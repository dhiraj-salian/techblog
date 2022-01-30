---
layout: post
title:  "Introduction to Cloud Native Fundamentals"
date:   2021-06-07 22:30:30 +0530
image: "assets/images/cover/surface-R8bY83YDXnY-unsplash.jpg"
categories: deploy kubernetes cloud-native
---

Cloud-native refers to set of practices that helps an organization **to build and manage applications at scale** using private, hybrid or public cloud providers. It also helps to increase feature velocity i.e. how quickly an organization can respond to changes and be agile.

Containers are closely associated with cloud-native terminology. Containers used to run a single application with all required dependencies. The main characteristics of containers are easy to manage, deploy and fast to recover. Therefore, Microservice-based architecture fits cloud native.

### Kubernetes Introduction

With the advent of containers came the need for a tool to manage containers. This need brought to limelight, container orchestration tools such as Dockerswarm, Apache Mesos and Kubernetes. However Kubernetes (derived from Borg, Google Open Source Project) took the lead and defined how containers should be managed, configured and deployed. Currently Kubernetes is part of CNCF (Cloud Native Computing Foundation), an organization that provides home to similar vendor-neutral open source projects.

Kubernetes automates configuration, management and scalability of application. Over time Kubernetes capabilities were extended via other tools for following functionalities:
  1. Runtime - for application execution environment
  2. Network - for application connectivity
  3. Storage - for application resources
  4. Service Mesh - for granular control of the traffic within cluster
  5. Logs and metrics - to construct observability stack
  6. Tracing - for building the full request journey

### Cloud Native Adoption

Before adopting an open source technology it is important to understand the business and technical perspective of the technology.

#### Business Perspective
  * Agility - quickly adapt to changing customer needs
  * Growth - growth in customer base
  * Service Availability - isolation of services which helps in quick recovery of failed services and minimize down time of the service.

#### Technical Perspective
 * Automation - construct pipeline to deploy without any human intervention
 * Orchestration - manage thousands of services with minimal effort
 * Observability - ability to debug each component independently