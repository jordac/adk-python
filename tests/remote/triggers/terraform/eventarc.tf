# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ---------------------------------------------------------------------------
# Eventarc: separate Pub/Sub topic as event source → Cloud Run /trigger/eventarc
# ---------------------------------------------------------------------------

# A dedicated topic that acts as the Eventarc event source.
# Publishing to this topic triggers the Eventarc → Cloud Run pipeline.
resource "google_pubsub_topic" "eventarc_source" {
  name    = "${local.name_prefix}-eventarc-${local.suffix}"
  project = var.project_id

  depends_on = [google_project_service.apis]
}

resource "google_eventarc_trigger" "trigger_test" {
  name     = "${local.name_prefix}-${local.suffix}"
  location = var.region
  project  = var.project_id

  matching_criteria {
    attribute = "type"
    value     = "google.cloud.pubsub.topic.v1.messagePublished"
  }

  transport {
    pubsub {
      topic = google_pubsub_topic.eventarc_source.id
    }
  }

  destination {
    cloud_run_service {
      service = data.google_cloud_run_v2_service.trigger_agent.name
      path    = "/apps/trigger_echo_agent/trigger/eventarc"
      region  = var.region
    }
  }

  service_account = google_service_account.eventarc_invoker.email

  depends_on = [
    google_project_iam_member.eventarc_event_receiver,
    google_project_service.apis,
  ]
}
