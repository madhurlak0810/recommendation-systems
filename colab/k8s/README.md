# Testing K8S
install awscli kubectl eksctl minikube

minikube start --driver=docker
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
minikube service colab-service
minikube stop

# TODO K8S
Could possibly include a readinessProbe and livenessProbe

kubectl port-forward service/<your-service-name> 8000:8000 --> use a specific port

minikube service colab-app --url --> get ip and port

kubectl apply -f k8s/ --> can do this instead

Look into ingress: https://www.reddit.com/r/kubernetes/comments/pa9jfg/im_a_newb_to_kubernetes_why_do_i_need/

🛠 Typical Setup Flow:
✅ Push Docker image to DockerHub (or ECR)

✅ Write Kubernetes YAMLs for each model/service

✅ Install kubectl, eksctl, and awscli

⬆️ Create EKS cluster (eksctl create cluster ...)

⬇️ Deploy your apps using kubectl apply -f your-deployment.yaml

🌐 Expose services (via LoadBalancer or Ingress) so they’re publicly accessible

📦 Communicate with the apps using requests or any other client