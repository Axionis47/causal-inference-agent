terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Uncomment to use GCS remote state
  # backend "gcs" {
  #   bucket = "your-terraform-state-bucket"
  #   prefix = "causal-orchestrator"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ─── Backend Cloud Run Service ───────────────────────────────────────────────
resource "google_cloud_run_v2_service" "backend" {
  name     = "causal-backend${var.environment_suffix}"
  location = var.region

  template {
    scaling {
      min_instance_count = var.backend_min_instances
      max_instance_count = var.backend_max_instances
    }

    containers {
      image = "${var.registry}/${var.project_id}/causal-backend:${var.image_tag}"

      ports {
        container_port = 8080
      }

      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }

      env {
        name  = "USE_FIRESTORE"
        value = "true"
      }

      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }

      env {
        name = "CLAUDE_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.claude_api_key.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "KAGGLE_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.kaggle_key.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "KAGGLE_USERNAME"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.kaggle_username.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.api_key.secret_id
            version = "latest"
          }
        }
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 10
        period_seconds        = 5
        failure_threshold     = 5
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
        period_seconds = 30
      }
    }

    service_account = google_service_account.backend.email
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}

# Allow unauthenticated access to backend (API key auth handled in app)
resource "google_cloud_run_v2_service_iam_member" "backend_public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.backend.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ─── Frontend Cloud Run Service ──────────────────────────────────────────────
resource "google_cloud_run_v2_service" "frontend" {
  name     = "causal-frontend${var.environment_suffix}"
  location = var.region

  template {
    scaling {
      min_instance_count = var.frontend_min_instances
      max_instance_count = var.frontend_max_instances
    }

    containers {
      image = "${var.registry}/${var.project_id}/causal-frontend:${var.image_tag}"

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}

resource "google_cloud_run_v2_service_iam_member" "frontend_public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.frontend.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ─── Service Account ─────────────────────────────────────────────────────────
resource "google_service_account" "backend" {
  account_id   = "causal-backend${var.environment_suffix}"
  display_name = "Causal Orchestrator Backend"
}

# Grant Firestore access
resource "google_project_iam_member" "backend_firestore" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.backend.email}"
}

# Grant Secret Manager access
resource "google_project_iam_member" "backend_secrets" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.backend.email}"
}

# Grant GCS access
resource "google_project_iam_member" "backend_gcs" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.backend.email}"
}
