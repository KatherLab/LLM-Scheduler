from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    router_host: str = Field(default="0.0.0.0", alias="ROUTER_HOST")
    router_port: int = Field(default=9000, alias="ROUTER_PORT")

    public_hostname: str = Field(default="127.0.0.1", alias="PUBLIC_HOSTNAME")
    database_url: str = Field(default="sqlite:////var/lib/vllm-router/router.db", alias="DATABASE_URL")

    slurm_partition: str | None = Field(default=None, alias="SLURM_PARTITION")
    slurm_account: str | None = Field(default=None, alias="SLURM_ACCOUNT")
    slurm_qos: str | None = Field(default=None, alias="SLURM_QOS")
    slurm_nodelist: str | None = Field(default=None, alias="SLURM_NODELIST")
    slurm_cpus_per_task: int = Field(default=32, alias="SLURM_CPUS_PER_TASK")

    # Logs: default to repo-local logs for dev
    vllm_log_dir: str = Field(default="./logs", alias="VLLM_LOG_DIR")

    sbatch_template_path: str = Field(
        default="/opt/vllm-swapper-router/templates/vllm_job.sh",
        alias="SBATCH_TEMPLATE_PATH",
    )

    total_gpus: int = Field(default=8, alias="TOTAL_GPUS")
    scheduler_submit_lead_seconds: int = Field(default=120, alias="SCHEDULER_SUBMIT_LEAD_SECONDS")

    allow_on_demand_start: bool = Field(default=False, alias="ALLOW_ON_DEMAND_START")
    on_demand_max_wait_seconds: int = Field(default=30, alias="ON_DEMAND_MAX_WAIT_SECONDS")

    # vLLM job behavior (fail fast + one retry)
    vllm_health_timeout_seconds: int = Field(default=180, alias="VLLM_HEALTH_TIMEOUT_SECONDS")
    vllm_max_retries: int = Field(default=2, alias="VLLM_MAX_RETRIES")
    vllm_retry_delay_seconds: int = Field(default=60, alias="VLLM_RETRY_DELAY_SECONDS")


settings = Settings()
