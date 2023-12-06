# setup aws
aws configure 
aws configure set region us-west-1    # if not already set

# create a bucket
# aws s3 mb <s3://bucket name>
aws s3 mb s3://bilateral-brain         

# sync 
# aws s3 sync <source> <target>
aws s3 sync ./ s3://bilateral-brain


# # see s3 contents
# aws s3 ls 
# aws s3 ls s3://bilateral-brain

# # cp s3 bucket to local
# aws s3 cp s3://bilateral-brain ./ --recursive