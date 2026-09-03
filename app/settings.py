from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    router_host: str = Field(default="0.0.0.0", alias="ROUTER_HOST")
    router_port: int = Field(default=9000, alias="ROUTER_PORT")

    public_hostname: str = Field(default="127.0.0.1", alias="PUBLIC_HOSTNAME")
    database_url: str = Field(default="sqlite:////var/lib/vllm-router/router.db", alias="DATABASE_URL")

    # --- Auth ---
    # "password" keeps the historical shared-password login; "ldap" binds
    # against FreeIPA and resolves group membership for RBAC.
    auth_mode: str = Field(default="password", alias="AUTH_MODE")
    auth_password: str = Field(default="changeme", alias="AUTH_PASSWORD")
    auth_secret_key: str = Field(default="", alias="AUTH_SECRET_KEY")
    auth_session_max_age_seconds: int = Field(default=86400, alias="AUTH_SESSION_MAX_AGE_SECONDS")

    # Break-glass admin using AUTH_PASSWORD. Keep enabled: an IPA outage must
    # not lock us out of the scheduler we would need to fix it.
    local_admin_enabled: bool = Field(default=True, alias="LOCAL_ADMIN_ENABLED")

    # --- LDAP / FreeIPA (only used when AUTH_MODE=ldap) ---
    ldap_url: str = Field(default="", alias="LDAP_URL")
    ldap_user_dn_template: str = Field(
        default="uid={username},cn=users,cn=accounts,dc=example,dc=com",
        alias="LDAP_USER_DN_TEMPLATE",
    )
    ldap_group_base_dn: str = Field(default="", alias="LDAP_GROUP_BASE_DN")
    ldap_group_filter: str = Field(
        default="(&(objectClass=groupOfNames)(member={user_dn}))",
        alias="LDAP_GROUP_FILTER",
    )
    ldap_start_tls: bool = Field(default=False, alias="LDAP_START_TLS")
    ldap_timeout_seconds: int = Field(default=5, alias="LDAP_TIMEOUT_SECONDS")
    ldap_group_cache_seconds: int = Field(default=300, alias="LDAP_GROUP_CACHE_SECONDS")

    # --- RBAC ---
    # Comma-separated LDAP group names.
    admin_groups: str = Field(default="llm-admins", alias="ADMIN_GROUPS")
    # Empty means any authenticated identity may create bookings.
    user_groups: str = Field(default="", alias="USER_GROUPS")
    # Per-pool stewards as "pool:group,pool:group". Superseded by cluster.yaml
    # once Phase 1 lands.
    pool_operators: str = Field(default="", alias="POOL_OPERATORS")

    # Which ClusterBackend implementation to use:
    #   "slurm_rest" — slurmrestd over JWT; no Slurm binaries or munge needed
    #   "slurm_cli"  — subprocess; must run on a host with the Slurm binaries
    #   "local"      — in-memory fake, for tests and laptop development
    cluster_backend: str = Field(default="slurm_cli", alias="CLUSTER_BACKEND")

    # --- slurmrestd (CLUSTER_BACKEND=slurm_rest) ---
    # Full versioned base URL, e.g. http://titan:6820/slurm/v0.0.42
    slurm_rest_url: str = Field(default="", alias="SLURM_REST_URL")
    # Token from `scontrol token lifespan=...`. These expire — see below.
    slurm_jwt: str = Field(default="", alias="SLURM_JWT")

    # --- Token renewal ---
    # static  : use SLURM_JWT as-is. Simple; expires and pauses scheduling.
    # file    : read from SLURM_TOKEN_FILE, refreshed by cron/systemd/k8s.
    #           Safest: the app never holds a credential that mints credentials.
    # command : run SLURM_TOKEN_COMMAND (or the SSH recipe below) to mint one.
    slurm_token_mode: str = Field(default="static", alias="SLURM_TOKEN_MODE")
    slurm_token_file: str = Field(default="", alias="SLURM_TOKEN_FILE")
    slurm_token_command: str = Field(default="", alias="SLURM_TOKEN_COMMAND")
    # Renew this long before the token's own `exp` claim.
    slurm_token_refresh_margin_seconds: float = Field(
        default=300.0, alias="SLURM_TOKEN_REFRESH_MARGIN_SECONDS")
    slurm_token_timeout_seconds: float = Field(
        default=30.0, alias="SLURM_TOKEN_TIMEOUT_SECONDS")
    slurm_token_lifespan_seconds: int = Field(
        default=3600, alias="SLURM_TOKEN_LIFESPAN_SECONDS")

    # SSH recipe: ssh <user>@<host> scontrol token lifespan=<n>
    # The remote authorized_keys entry SHOULD pin a forced command so a stolen
    # key can only mint tokens, never open a shell. See config/example.env.
    slurm_token_ssh_host: str = Field(default="", alias="SLURM_TOKEN_SSH_HOST")
    slurm_token_ssh_user: str = Field(default="", alias="SLURM_TOKEN_SSH_USER")
    slurm_token_ssh_key: str = Field(default="", alias="SLURM_TOKEN_SSH_KEY")
    slurm_token_ssh_port: int = Field(default=0, alias="SLURM_TOKEN_SSH_PORT")
    slurm_token_ssh_known_hosts: str = Field(
        default="", alias="SLURM_TOKEN_SSH_KNOWN_HOSTS")
    # Account to submit as. Important when the JWT belongs to a privileged user:
    # without it, jobs run as the token's owner (often root).
    slurm_rest_user: str | None = Field(default=None, alias="SLURM_REST_USER")
    slurm_rest_timeout_seconds: float = Field(default=30.0, alias="SLURM_REST_TIMEOUT_SECONDS")
    slurm_rest_verify_tls: bool = Field(default=True, alias="SLURM_REST_VERIFY_TLS")

    # Backend used only for `sbatch --test-only` start estimates, which have no
    # slurmrestd equivalent. Set to "slurm_cli" when the router sits on a host
    # with the Slurm binaries; leave empty to report "start time unknown".
    estimate_backend: str = Field(default="", alias="ESTIMATE_BACKEND")

    slurm_partition: str | None = Field(default=None, alias="SLURM_PARTITION")
    slurm_account: str | None = Field(default=None, alias="SLURM_ACCOUNT")
    slurm_qos: str | None = Field(default=None, alias="SLURM_QOS")
    slurm_nodelist: str | None = Field(default=None, alias="SLURM_NODELIST")
    slurm_cpus_per_task: int = Field(default=16, alias="SLURM_CPUS_PER_TASK")

    # Application log level. uvicorn configures only its own loggers, so
    # without this the app's own INFO lines (token renewal, inventory refresh,
    # leader changes) are silently dropped — exactly the ones you want when
    # something is wrong.
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Logs: default to repo-local logs for dev.
    # This one is OURS — lifecycle.log, written by this process.
    vllm_log_dir: str = Field(default="./logs", alias="VLLM_LOG_DIR")

    # Where Slurm writes each job's stdout/stderr. This path is resolved BY THE
    # COMPUTE NODE, not by us, so it must exist cluster-wide on a shared
    # filesystem. A container-local path like /app/logs is not merely useless
    # here — Slurm cannot open the output file and the job dies at launch, so
    # the failure has no log to explain itself.
    #
    # Empty falls back to VLLM_LOG_DIR, which is the historical behaviour and
    # correct whenever the router runs on the cluster's own filesystem.
    job_log_dir: str = Field(default="", alias="JOB_LOG_DIR")

    # The same directory as *this* process sees it, when it is mounted
    # somewhere else (a container bind, a different NFS mountpoint). Only the
    # log-viewing endpoint uses it. Empty means "same path as the nodes see".
    job_log_dir_local: str = Field(default="", alias="JOB_LOG_DIR_LOCAL")

    sbatch_template_path: str = Field(
        default="/opt/vllm-swapper-router/templates/vllm_job.sh",
        alias="SBATCH_TEMPLATE_PATH",
    )

    # --- Apptainer image management ---
    # The shared directory compute nodes read .sif images from. It must be
    # mounted here too: listing and deleting are plain filesystem operations,
    # and without it the images UI reports that rather than pretending the
    # cluster has no images. Empty disables image management entirely.
    apptainer_image_dir: str = Field(default="", alias="APPTAINER_IMAGE_DIR")

    # Working area for `apptainer build` — unpacked layers run several times
    # the size of the finished .sif. Empty means `<images>/../build-tmp`.
    # Do not point this at a RAM-backed /tmp.
    image_build_scratch: str = Field(default="", alias="IMAGE_BUILD_SCRATCH")

    image_build_template_path: str = Field(
        default="./templates/apptainer_build.sh",
        alias="IMAGE_BUILD_TEMPLATE_PATH",
    )
    # A multi-GB image pull plus mksquashfs; generous because the cost of it
    # timing out at 95% is another full build.
    image_build_time_limit: str = Field(default="02:00:00", alias="IMAGE_BUILD_TIME_LIMIT")
    image_build_cpus: int = Field(default=8, alias="IMAGE_BUILD_CPUS")

    # Optional registry credentials, for private sources such as nvcr.io.
    # These reach the build job through its environment, which on this cluster
    # is readable by the service account the jobs run as.
    image_registry_username: str = Field(default="", alias="IMAGE_REGISTRY_USERNAME")
    image_registry_password: str = Field(default="", alias="IMAGE_REGISTRY_PASSWORD")

    # --- High availability ---
    # With HA on, every replica serves proxy traffic but only the leader runs
    # the background workers — two instances submitting for the same booking
    # would double-submit. Off (the default) means single-instance behaviour.
    ha_enabled: bool = Field(default=False, alias="HA_ENABLED")

    total_gpus: int = Field(default=8, alias="TOTAL_GPUS")
    scheduler_submit_lead_seconds: int = Field(default=120, alias="SCHEDULER_SUBMIT_LEAD_SECONDS")

    vllm_api_key: str = Field(default="secret", alias="VLLM_API_KEY")

    schedule_api_key: str = Field(default="", alias="SCHEDULE_API_KEY")

    allow_on_demand_start: bool = Field(default=False, alias="ALLOW_ON_DEMAND_START")
    on_demand_max_wait_seconds: int = Field(default=30, alias="ON_DEMAND_MAX_WAIT_SECONDS")

    # vLLM job behavior (fail fast + one retry)
    vllm_health_timeout_seconds: int = Field(default=800, alias="VLLM_HEALTH_TIMEOUT_SECONDS")
    vllm_max_retries: int = Field(default=1, alias="VLLM_MAX_RETRIES")
    vllm_retry_delay_seconds: int = Field(default=10, alias="VLLM_RETRY_DELAY_SECONDS")

    # Slurm email notifications (optional)
    # Set SLURM_MAIL_USER to an email address to receive job notifications.
    # SLURM_MAIL_TYPE controls which events trigger emails.
    # Valid values: NONE, BEGIN, END, FAIL, REQUEUE, ALL, TIME_LIMIT, TIME_LIMIT_90, etc.
    # Multiple values can be comma-separated, e.g. "BEGIN,END,FAIL"
    slurm_mail_user: str | None = Field(default=None, alias="SLURM_MAIL_USER")
    slurm_mail_type: str = Field(default="FAIL,END,TIME_LIMIT", alias="SLURM_MAIL_TYPE")

    @model_validator(mode="after")
    def _resolve_log_dirs(self) -> "Settings":
        """Fill in the job-log paths so callers never repeat the fallback chain."""
        if not self.job_log_dir:
            self.job_log_dir = self.vllm_log_dir
        if not self.job_log_dir_local:
            self.job_log_dir_local = self.job_log_dir
        return self


settings = Settings()
