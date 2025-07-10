import pandas as pd
import json
import argparse
import os
import openai
from logger import setup_logger
from typing import Optional, Literal
import time

# 设置日志
logger = setup_logger(__name__)


class GPTFineTuner:
    def __init__(self, api_key: str):
        """Initialize the fine-tuner with OpenAI API key."""
        self.client = openai.OpenAI(api_key=api_key)
        self.training_file_id = None
        self.validation_file_id = None
        self.job_id = None
        self.model_id = None

    def upload_file(self, file_path: str, purpose: Literal["fine-tune", "assistants", "batch", "vision", "user_data", "evals"] = "fine-tune") -> Optional[str]:
        """
        Upload file to OpenAI and return file ID
        """
        logger.info(f"上传 {file_path} 至OpenAI...")

        try:
            with open(file_path, "rb") as f:
                response = self.client.files.create(file=f, purpose=purpose)

            file_id = response.id
            logger.info(f"文件上传成功, 文件ID: {file_id}")
            return file_id

        except Exception as e:
            logger.error(f"文件上传失败: {e}")
            return None


    def create_fine_tune_job(
        self,
        training_file_id: str,
        validation_file_id: Optional[str] = None,
        model: str = "gpt-4.1-mini-2025-04-14",
        n_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate_multiplier: Optional[float] = None,
    ) -> Optional[str]:
        """
        Create a fine-tune job
        """
        logger.info("创建微调任务...")

        try:
            # 创建超参数
            hyperparameters = {}

            logger.info(f"训练轮数: {n_epochs}")
            hyperparameters["n_epochs"] = n_epochs

            if batch_size is not None:
                logger.info(f"批量大小: {batch_size}")
                hyperparameters["batch_size"] = batch_size

            if learning_rate_multiplier is not None:
                logger.info(f"学习率乘数: {learning_rate_multiplier}")
                hyperparameters["learning_rate_multiplier"] = learning_rate_multiplier

            # 创建基础参数
            job_params = {
                "training_file": training_file_id,
                "model": model,
                "hyperparameters": hyperparameters,
            }

            logger.info(f"训练文件ID: {training_file_id}")
            logger.info(f"模型: {model}")

            # 如果有验证文件，添加到参数中
            if validation_file_id is not None:
                logger.info(f"验证文件ID: {validation_file_id}")
                job_params["validation_file"] = validation_file_id

            logger.info(f"用以上参数请求创建微调任务...")
            response = self.client.fine_tuning.jobs.create(**job_params)

            job_id = response.id
            logger.info(f"微调任务创建成功, 任务ID: {job_id}")
            return job_id

        except Exception as e:
            logger.error(f"创建微调任务失败: {e}")
            return None

    def monitor_training(self, job_id: str, check_interval: int = 20) -> Optional[str]:
        """
        Monitor progress of fine-tuning job with detailed metrics
        """
        logger.info(f"监督微调任务 {job_id}...")
        
        # Track metrics over time
        training_history = []
        best_val_loss = float('inf')
        patience_counter = 0
        patience_limit = 5  # Stop if no improvement for 5 checks

        try:
            while True:
                job = self.client.fine_tuning.jobs.retrieve(job_id)
                status = job.status
                
                # Basic status info
                logger.info(f"状态: {status}")
                
                # Token progress
                if hasattr(job, "trained_tokens") and job.trained_tokens:
                    logger.info(f"训练token数: {job.trained_tokens:,}")
                
                # Get training metrics from events
                events = self.client.fine_tuning.jobs.list_events(job_id, limit=50)
                latest_metrics = self._extract_latest_metrics(events)
                
                if latest_metrics:
                    step = latest_metrics.get('step', 0)
                    train_loss = latest_metrics.get('train_loss')
                    valid_loss = latest_metrics.get('valid_loss')
                    train_acc = latest_metrics.get('train_accuracy')
                    valid_acc = latest_metrics.get('valid_accuracy')
                    learning_rate = latest_metrics.get('learning_rate')
                    
                    # Log current metrics
                    logger.info("=== 训练指标 ===")
                    if train_loss is not None:
                        logger.info(f"训练损失: {train_loss:.4f}")
                    if valid_loss is not None:
                        logger.info(f"验证损失: {valid_loss:.4f}")
                    if train_acc is not None:
                        logger.info(f"训练准确率: {train_acc:.3f}")
                    if valid_acc is not None:
                        logger.info(f"验证准确率: {valid_acc:.3f}")
                    if learning_rate is not None:
                        logger.info(f"学习率: {learning_rate:.6f}")
                    
                    # Store history
                    training_history.append(latest_metrics)
                    
                    # Analyze training health
                    self._analyze_training_health(training_history, latest_metrics)
                    
                    # Early stopping logic
                    if valid_loss is not None:
                        if valid_loss < best_val_loss:
                            best_val_loss = valid_loss
                            patience_counter = 0
                            logger.info("验证损失改善!")
                        else:
                            patience_counter += 1
                            logger.info(f"验证损失未改善 ({patience_counter}/{patience_limit})")
                            
                            if patience_counter >= patience_limit:
                                logger.warning("考虑提前停止 - 验证损失未持续改善")
                    
                    logger.info("=" * 20)

                # Check final status
                if status == "succeeded":
                    self.model_id = job.fine_tuned_model
                    logger.info(f"训练完成! 模型ID: {self.model_id}")
                    self._log_final_summary(training_history)
                    return self.model_id
                elif status == "failed":
                    logger.error(f"训练失败, 错误: {job.error}")
                    return None
                elif status == "cancelled":
                    logger.info("训练取消")
                    return None

                time.sleep(check_interval)

        except KeyboardInterrupt:
            logger.info("停止监督，取消微调任务")
            self.cancel_training(job_id)
            return None
        except Exception as e:
            logger.error(f"监督失败: {e}")
            return None

    def _extract_latest_metrics(self, events) -> dict:
        """Extract latest training metrics from events"""
        latest_metrics = {}
        
        for event in events.data:
            if event.type == "metrics":
                data = event.data
                if isinstance(data, dict):
                    latest_metrics.update(data)
        
        return latest_metrics

    def _analyze_training_health(self, history: list, current: dict):
        """Analyze training health and provide warnings"""
        if len(history) < 3:
            return
        
        train_losses = [h.get('train_loss') for h in history[-3:] if h.get('train_loss')]
        valid_losses = [h.get('valid_loss') for h in history[-3:] if h.get('valid_loss')]
        
        if len(train_losses) >= 2 and len(valid_losses) >= 2:
            # Check for overfitting
            train_trend = train_losses[-1] - train_losses[0]
            valid_trend = valid_losses[-1] - valid_losses[0]
            
            if train_trend < -0.01 and valid_trend > 0.01:
                logger.warning("可能过拟合: 训练损失下降但验证损失上升")
            
            # Check for underfitting
            if abs(train_trend) < 0.001 and abs(valid_trend) < 0.001:
                logger.warning("可能欠拟合: 损失变化很小")
        
        # Check learning rate
        lr = current.get('learning_rate')
        if lr and lr < 1e-6:
            logger.warning("学习率过低，可能影响训练效果")

    def _log_final_summary(self, history: list):
        """
        Log training summary
        """
        if not history:
            return
        
        logger.info("\n=== 训练总结 ===")
        final_metrics = history[-1]
        
        if final_metrics.get('train_loss'):
            logger.info(f"最终训练损失: {final_metrics['train_loss']:.4f}")
        if final_metrics.get('valid_loss'):
            logger.info(f"最终验证损失: {final_metrics['valid_loss']:.4f}")
        if final_metrics.get('valid_accuracy'):
            logger.info(f"最终验证准确率: {final_metrics['valid_accuracy']:.3f}")
        
        logger.info(f"总训练步数: {len(history)}")
        logger.info("=" * 20)


    def cancel_training(self, job_id: str) -> bool:
        """
        Cancel a running fine-tuning job
        """
        try:
            cancelled_job = self.client.fine_tuning.jobs.cancel(job_id)
            logger.info(f"训练任务已取消: {job_id}")
            logger.info(f"状态: {cancelled_job.status}")
            return True
        except Exception as e:
            logger.error(f"取消失败: {e}")
            return False


    def ask_model(self, model_id: str, system_prompt: str, test_message: str, max_tokens: int = 200, temperature: float = 0.0):
        """
        Ask fine-tuned model with a test message
        """
        try:
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": test_message}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            result = response.choices[0].message.content
            logger.info(f"Model response: {result}")
            return result

        except Exception as e:
            logger.error(f"Error testing model: {e}")
            return None

    def list_available_models(self):
        """
        List available models from OpenAI API, but only return those matching the allowed set for this app.
        Returns a list of dicts: [{"id": ..., "object": ...}, ...]
        """
        allowed_ids = [
            "gpt-4.1-mini-2025-04-14",
            "o4-mini-2025-04-16",
            "o3-2025-04-16",
            "o3-pro-2025-06-10",
            "o3-mini-2025-01-31",
            "gpt-4.1-2025-04-14",
            "gpt-4o-2024-11-20",
            "gpt-4.1-nano-2025-04-14",
            "gpt-4o-mini-2024-07-18",
        ]
        try:
            response = self.client.models.list()
            models = response.data if hasattr(response, 'data') else response
            # logger.info(f"models: {[getattr(k, 'id', None) for k in models]}")
            # Build a dict for fast lookup
            model_dict = {getattr(m, 'id', None): m for m in models if getattr(m, 'id', None) in allowed_ids}
            # Warn if any allowed_ids are missing
            missing = [mid for mid in allowed_ids if mid not in model_dict]
            if missing:
                logger.warning(f"以下模型在OpenAI返回的列表中缺失: {missing}")
            filtered = []
            for model_id in allowed_ids:
                m = model_dict.get(model_id)
                if m:
                    filtered.append({"id": model_id, "object": getattr(m, 'object', '')})
            logger.info(f"可用模型: {[m['id'] for m in filtered]}")
            return filtered
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return []
