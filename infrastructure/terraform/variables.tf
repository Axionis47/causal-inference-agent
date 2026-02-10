variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Run services"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment (staging or production)"
  type        = string
  default     = "staging"

  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "Environment must be 'staging' or 'production'."
  }
}

variable "environment_suffix" {
  description = "Suffix for resource names (e.g., '-staging' or empty for production)"
  type        = string
  default     = "-staging"
}

variable "image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest"
}

variable "registry" {
  description = "Container registry URL"
  type        = string
  default     = "gcr.io"
}

variable "backend_min_instances" {
  description = "Minimum backend instances"
  type        = number
  default     = 0
}

variable "backend_max_instances" {
  description = "Maximum backend instances"
  type        = number
  default     = 10
}

variable "frontend_min_instances" {
  description = "Minimum frontend instances"
  type        = number
  default     = 0
}

variable "frontend_max_instances" {
  description = "Maximum frontend instances"
  type        = number
  default     = 5
}
