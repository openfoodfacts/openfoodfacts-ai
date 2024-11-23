export ip_vm=54-209-201-47
ssh -i "~/.ssh/linsight.pem" ec2-user@ec2-$ip_vm.compute-1.amazonaws.com

bash << EOF
rsync -avzhP --stats -e "ssh -i ~/.ssh/linsight.pem" --exclude={'data','.git','images','serve'}  ~/PycharmProjects/openfoodfacts-ai ec2-user@ec2-$ip_vm.compute-1.amazonaws.com:~/
EOF