# setup aws
aws configure 
aws configure set region us-west-1    # if not already set

# create a bucket
# aws s3 mb <bucket name>
aws s3 mb bilateral-brain         

# sync 
# aws s3 sync <source> <target>
aws s3 sync ./ s3://bilateral-brain