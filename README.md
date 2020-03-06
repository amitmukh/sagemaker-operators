This repository contains code and config files supporting SageMaker Operator demo. and credit goes to shashankprasanna

### Amazon SageMaker Operators for Kubernetes and how to use it:
Amazon SageMaker Operators for Kubernetes is implemented as a custom resource in Kubernetes and enables Kubernetes to invoke Amazon SageMaker functionality. Below, I’ll provide step-by-step instructions for implementing each of these use cases:

Use case 1: Distributed training with TensorFlow, PyTorch, MXNet and other frameworks
Use case 2: Distributed training with a custom container
Use case 3: Hyperparameter optimization at-scale with TensorFlow
Use case 4: Hosting an inference endpoint with BYO model  

To follow along, I assume you have an AWS account, and AWS CLI tool, Kubectl, AWS IAM Authenticator installed on your host machine. If not then use the below link to install the same:

- AWS account Creation: https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/
- AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv1.html
- Kubectl (Version 1.13 or higher) - https://docs.aws.amazon.com/eks/latest/userguide/getting-started-eksctl.html
- AWS IAM Authenticator for Kubernetes - https://docs.aws.amazon.com/eks/latest/userguide/install-aws-iam-authenticator.html

## Setup

Let’s start by spinning up a Kubernetes cluster. With the eksctl CLI tool, all it takes is a simple command and 15 mins of your time for a very simple cluster with a couple of nodes.

## Create a Kubernetes cluster

eksctl create cluster \
    --name sm-operator-demo \
    --version 1.14 \
    --region us-west-2 \
    --nodegroup-name test-nodes \
    --node-type c5.xlarge \
    --nodes 1 \
    --node-volume-size 50 \
    --node-zones us-west-2a \
    --timeout=40m \
    --auto-kubeconfig

## Install Amazon SageMaker Operators for Kubernetes

Once the cluster is up and running, follow the instructions in the user guide to install Amazon SageMaker Operators for Kubernetes. You can also refer to this helpful blog post to guide your installation process: [Introducing Amazon SageMaker Operators for Kubernetes](https://https://aws.amazon.com/blogs/machine-learning/introducing-amazon-sagemaker-operators-for-kubernetes/)

To verify installation run

kubectl get crd | grep sagemaker

You should get an output that looks something like this:

batchtransformjobs.sagemaker.aws.amazon.com                 2020-02-29T21:21:24Z
endpointconfigs.sagemaker.aws.amazon.com                    2020-02-29T21:21:24Z
hostingdeployments.sagemaker.aws.amazon.com                 2020-02-29T21:21:24Z
hyperparametertuningjobs.sagemaker.aws.amazon.com         2020-02-29T21:21:24Z
models.sagemaker.aws.amazon.com                                2020-02-29T21:21:24Z
trainingjobs.sagemaker.aws.amazon.com                           2020-02-29T21:21:24Z

These are all the tasks you can perform on Amazon SageMaker using the Amazon SageMaker Operators for Kubernetes, and we’ll take a closer look at (1) training jobs (2) hyperparameter tuning jobs (3) hosting deployments.

## Download examples from GitHub

Download training scripts, config files and Jupyter notebooks to your host machine.

git clone https://github.com/amitmukh/kubernetes-sagemaker-demos.git

## Download training dataset and upload to Amazon S3

cd kubernetes-sagemaker-demos/0-upload-dataset-s3

Note: TensorFlow must be installed on the host machine to download the dataset and convert into the TFRecord format. So instead of using local laptop you can use Sagemaker to do this part.

## Use case 1: Distributed training with TensorFlow, PyTorch, MXNet and other frameworks

If you’re new to Amazon SageMaker, one of its nice features when using popular frameworks such as TensorFlow, PyTorch, MXNet, XGBoost and others is that you don’t have to worry about building custom containers with your code in it and pushing it to a container registry. Amazon SageMaker can automatically download any training scripts and dependencies into a framework container and run it at scale for you. So you just have to version and manage your training scripts and don’t have to deal with containers at all. With Amazon SageMaker Operators for Kubernetes, you can still get the same experience.
Navigate to the directory with the 1st example:

cd kubernetes-sagemaker-demos/1-tf-dist-training-training-script/
ls -1

Output:

cifar10-multi-gpu-horovod-sagemaker.py
k8s-sm-dist-training-script.yaml
model_def.py
upload_source_to_s3.ipynb

The two python files in this directory cifar10-multi-gpu-horovod-sagemaker.py and model_def.py are TensorFlow training scripts that implement Horovod API for distributed training.

Run through upload_source_to_s3.ipynb to create a tar file with the training scripts and upload it to the specified Amazon S3 bucket.

k8s-sm-dist-training-script.yaml a config file that when applied using kubectl kicks of a distributed training job. Open it in your favorite text editor to take a closer look.

First you’ll notice that kind: TrainingJob. This suggests that you’ll submit an Amazon SageMaker training job.

Under hyperParameters, specify the hyperparameters that cifar10-multi-gpu-horovod-sagemaker.py can accept as inputs.

Specify additional parameters for distributed training:
- sagemaker_program — cifar10-multi-gpu-horovod-sagemaker.py TensorFlow training script that implements Horovod API for distributed training
- sagemaker_submit_directory — location on Amazon S3 where training scripts are located
- sagemaker_mpi_enabled and sagemaker_mpi_custom_mpi_options — enable MPI communication for distributed training
- sagemaker_mpi_num_of_processes_per_host — set to the number of GPUs on the requested instance. For p3dn.24xlarge instance with 8 GPUs set this value to 8.

Specify the deep learning framework container by selecting the appropriate container from here:
https://docs.aws.amazon.com/dlami/latest/devguide/deep-learning-containers-images.html

Amazon SageMaker will automatically download the training scripts specified under sagemaker_submit_directory into the container instantiated from trainingImage.

Under resource config specify how many instances or nodes you want to run this multi-node training on. The above config file specifies that it’ll run distributed training on 32 GPUs.

Finally, specify the dataset location on Amazon S3. This should be the same bucket name you chose when running upload_source_to_s3.ipynb Jupyter notebook to upload the training dataset.

To start distributed training, run:

kubectl apply -f k8s-sm-dist-training-script.yaml

Output:

trainingjob.sagemaker.aws.amazon.com/k8s-sm-dist-training-script created

To get the training job information run:

kubectl get trainingjob

Output:

NAME                          STATUS       SECONDARY-STATUS   CREATION-TIME          SAGEMAKER-JOB-NAME
k8s-sm-dist-training-script   InProgress   Starting           2020-03-06T17:56:37Z   k8s-sm-dist-training-script-d31cc6b35fd311ea923f0eff5eeecd03

Now navigate to AWS Console > Amazon SageMaker > Training jobs
And you’ll see that a new training job with the same name as the output of kubectl get trainingjob

To view the training logs, click on the training job in the console and click on “View Logs” under the Monitor section. This will take you to CloudWatch where you can view the training logs.

Alternatively, if you installed [smlogs](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_operators_for_kubernetes.html#install-the-amazon-sagemaker-logs-kubectl-plugin) plugin, then you can run the following to view logs using kubectl:

kubectl smlogs trainingjob k8s-sm-dist-training-script
