# ─── Secret Manager Secrets ─────────────────────────────────────────────────
# NOTE: Secret values must be set manually via GCP Console or gcloud CLI.
# Terraform manages the secret resources; you add the versions.
#
# Example:
#   echo -n "your-api-key" | gcloud secrets versions add claude-api-key --data-file=-

resource "google_secret_manager_secret" "claude_api_key" {
  secret_id = "claude-api-key"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "kaggle_key" {
  secret_id = "kaggle-key"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "kaggle_username" {
  secret_id = "kaggle-username"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "api_key" {
  secret_id = "api-key"

  replication {
    auto {}
  }
}
