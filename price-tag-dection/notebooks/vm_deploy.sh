export vm=basla01@34.79.123.28
ssh -i "~/.ssh/id_rsa" $vm

bash << EOF
rsync -avzhP --stats -e "ssh -i ~/.ssh/id_rsa" --exclude={'data','.git','images','serve'}  ~/PycharmProjects/openfoodfacts-ai $vm:~/
EOF