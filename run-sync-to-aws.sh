# setup aws
aws configure 
aws configure set region us-west-1    # if not already set

# create a bucket
# aws s3 mb <s3://bucket name>
aws s3 mb s3://bilateral-brain         

# sync 
# aws s3 sync <source> <target>
aws s3 sync ./ s3://bilateral-brain