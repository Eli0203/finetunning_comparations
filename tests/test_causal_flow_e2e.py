"""End-to-end tests for Phase 5 causal training integration."""

import unittest
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import TrainerState, TrainerControl, TrainingArguments

from src.finetuner.causal_engine import CausalMonteCLoRAEngine
from src.finetuner.causal_training_orchestrator import CausalTrainingOrchestrator
from src.settings.settings import CausalTrainingConfig
from src.utils.causal_sampler import CausalWeightSampler
from src.utils.async_sampler import BackgroundSampler
from src.utils.multiprocessing import RingBuffer


class TinyDataset(Dataset):
    """Minimal dataset for e2e training flow tests."""

    def __init__(self, size: int = 32, seq_len: int = 8, vocab_size: int = 100):
        self.input_ids = torch.randint(0, vocab_size, (size, seq_len))
        self.attention_mask = torch.ones(size, seq_len, dtype=torch.long)
        self.token_type_ids = torch.zeros(size, seq_len, dtype=torch.long)
        self.labels = torch.randint(0, 2, (size,), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "labels": self.labels[idx],
        }


class TinyModel(nn.Module):
    """Small model returning HuggingFace-like outputs with loss."""

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 16, n_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        emb = self.embedding(input_ids).mean(dim=1)
        logits = self.classifier(emb)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return type("Output", (), {"loss": loss, "logits": logits})()


class DummyTrainer:
    """Trainer mock that executes callbacks and tracks state."""

    def __init__(self, model, train_loader, eval_loader, max_steps: int = 20):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.max_steps = max_steps
        self.callbacks = []
        self.state = type("TrainerStateObj", (), {"best_metric": 0.0})()

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def train(self):
        args = TrainingArguments(output_dir="./tmp_test_output", per_device_train_batch_size=4)
        control = TrainerControl()
        state = TrainerState()

        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        global_step = 0
        for _ in range(self.max_steps):
            for batch in self.train_loader:
                out = self.model(**batch)
                if out.loss is not None:
                    opt.zero_grad(set_to_none=True)
                    out.loss.backward()
                    opt.step()

                state.global_step = global_step
                for callback in self.callbacks:
                    callback.on_step_end(args, state, control)

                global_step += 1
                if global_step >= self.max_steps:
                    break

            if global_step >= self.max_steps:
                break

        self.state.best_metric = 0.75
        return {"train_loss": float(out.loss.detach().item())}

    def get_eval_dataloader(self):
        return self.eval_loader


class TestCausalFlowE2E(unittest.TestCase):
    """Phase 5 integration and validation tests."""

    def setUp(self):
        self.model = TinyModel()
        self.train_loader = DataLoader(TinyDataset(size=24), batch_size=4)
        self.eval_loader = DataLoader(TinyDataset(size=16), batch_size=4)

        self.lora_engine = {"name": "stub-lora-engine"}
        self.causal_engine = CausalMonteCLoRAEngine(self.lora_engine, causal_threshold=0.0, sample_budget=100)
        self.causal_sampler = CausalWeightSampler(self.causal_engine, self.model, device="cpu")

    def test_e2e_orchestrator_pipeline(self):
        trainer = DummyTrainer(self.model, self.train_loader, self.eval_loader, max_steps=10)
        config = CausalTrainingConfig(
            total_causal_budget=100,
            async_max_steps=5,
            apply_interval=2,
            device="cpu",
            enable_warmup=False,
        )

        orchestrator = CausalTrainingOrchestrator(
            self.lora_engine,
            self.causal_engine,
            trainer,
            self.causal_sampler,
            config,
        )

        orchestrator.prepare(self.model, self.train_loader)
        result = orchestrator.run_training()
        diagnostics = orchestrator.get_diagnostics()

        self.assertIn("train_loss", result)
        self.assertIn("causal_summary", diagnostics)
        self.assertIn("budget_utilization", diagnostics)
        self.assertIn("training_metrics", diagnostics)
        self.assertEqual(diagnostics["state"], orchestrator.COMPLETED)

    def test_warmup_state_has_loss_trajectory(self):
        trainer = DummyTrainer(self.model, self.train_loader, self.eval_loader, max_steps=6)
        config = CausalTrainingConfig(
            total_causal_budget=100,
            async_max_steps=5,
            apply_interval=2,
            device="cpu",
            enable_warmup=True,
            warmup_steps=3,
        )

        orchestrator = CausalTrainingOrchestrator(
            self.lora_engine,
            self.causal_engine,
            trainer,
            self.causal_sampler,
            config,
        )

        orchestrator.prepare(self.model, self.train_loader)
        orchestrator.run_training()
        warmup_state = self.causal_engine.get_warmup_state()

        self.assertTrue(warmup_state["enabled"])
        self.assertGreaterEqual(warmup_state["steps"], 1)
        self.assertGreaterEqual(len(warmup_state["loss_trajectory"]), 1)

    def test_marginal_likelihood_is_finite(self):
        trainer = DummyTrainer(self.model, self.train_loader, self.eval_loader, max_steps=6)
        config = CausalTrainingConfig(
            total_causal_budget=100,
            async_max_steps=5,
            apply_interval=2,
            device="cpu",
            enable_warmup=True,
            warmup_steps=2,
        )

        orchestrator = CausalTrainingOrchestrator(
            self.lora_engine,
            self.causal_engine,
            trainer,
            self.causal_sampler,
            config,
        )

        orchestrator.prepare(self.model, self.train_loader)
        orchestrator.run_training()
        mml = self.causal_engine.get_causal_summary()["marginal_likelihood"]

        self.assertIsNotNone(mml)
        self.assertTrue(torch.isfinite(torch.tensor(mml)).item())

    def test_memory_pipeline_objects_exist(self):
        trainer = DummyTrainer(self.model, self.train_loader, self.eval_loader, max_steps=4)
        config = CausalTrainingConfig(
            total_causal_budget=100,
            async_max_steps=3,
            apply_interval=2,
            device="cpu",
        )

        orchestrator = CausalTrainingOrchestrator(
            self.lora_engine,
            self.causal_engine,
            trainer,
            self.causal_sampler,
            config,
        )

        orchestrator.prepare(self.model, self.train_loader)
        self.assertIsNotNone(orchestrator.buffer)
        self.assertIsNotNone(orchestrator.async_sampler)
        self.assertIsNotNone(orchestrator.weight_applier)
        self.assertIsNotNone(orchestrator.budget_monitor)

    def test_async_sampler_compatible_with_ring_buffer(self):
        ring_buffer = RingBuffer(size=15)
        sampler = BackgroundSampler(
            buffer=ring_buffer,
            model=self.model,
            max_steps=2,
            causal_sampler=self.causal_sampler,
        )

        sampler.start()

        latest = None
        deadline = time.time() + 3.0
        while latest is None and time.time() < deadline:
            latest = ring_buffer.get_latest()
            if latest is None:
                time.sleep(0.05)

        sampler.stop()
        sampler.raise_if_failed()
        status = sampler.get_status()

        self.assertIsNone(status["last_error"])
        self.assertGreaterEqual(status["metrics"]["generated_batches"], 0)
        if latest is not None:
            self.assertIsInstance(latest, dict)

    def test_fail_closed_mml_continues_training(self):
        trainer = DummyTrainer(self.model, self.train_loader, self.eval_loader, max_steps=6)
        config = CausalTrainingConfig(
            total_causal_budget=100,
            async_max_steps=3,
            apply_interval=2,
            device="cpu",
        )

        orchestrator = CausalTrainingOrchestrator(
            self.lora_engine,
            self.causal_engine,
            trainer,
            self.causal_sampler,
            config,
        )

        orchestrator.prepare(self.model, self.train_loader)
        orchestrator.causal_engine.validate_marginal_likelihood = lambda *args, **kwargs: None

        result = orchestrator.run_training()
        diagnostics = orchestrator.get_diagnostics()

        self.assertIn("train_loss", result)
        self.assertEqual(diagnostics["state"], orchestrator.COMPLETED)
        self.assertGreaterEqual(
            diagnostics["weight_application_metrics"].get("skip_next_apply_requests", 0),
            1,
        )


if __name__ == "__main__":
    unittest.main()
