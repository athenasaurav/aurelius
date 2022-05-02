# Deployment Guide

## Install and upgrade PIP
```
sudo apt update
sudo apt install python3-pip
```
## Create Virtual Env
```
pip3 install virtualenv
virtualenv test
source test/bin/activate
```
## Prepare for Elastic Search
```
apt-get update -y
apt-get upgrade -y

apt-get install apt-transport-https ca-certificates gnupg2 -y
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | apt-key add -
sh -c 'echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" > /etc/apt/sources.list.d/elastic-7.x.list'
```
## Install Elastic search
```
apt-get update -y
apt-get install elasticsearch -y
```
## Start Elastic Search
```
systemctl start elasticsearch
systemctl enable elasticsearch
```
Note : Sometimes this work on Linode : ```systemctl restart elasticsearch```

## Test Elastic Search

```curl -X GET "localhost:9200"```

## Git Clone Aurelius
```
git clone https://github.com/athenasaurav/aurelius.git
cd aurelius
```

## Install requirments
```pip3 install -r requirements.txt```

## Solve Milvus Error
```
pip3 uninstall pymilvus
pip3 install pymilvus==1.1.2
```
# Run api.py
```
python3 api.py
```
