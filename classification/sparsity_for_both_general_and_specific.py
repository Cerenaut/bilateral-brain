import yaml
#generate new config so that wa can set k per_k. fine or coarse and resnet ot inverse resnet
def generate_config(class000,k000,per_k000,resnet_iden):

    with open('./configs/config.yaml','r') as f:
        b=yaml.safe_load(f)
    
        
        #change dataset 12
        b["dataset"]["train_dir"]="../datasets/CIFAR100/train/"+class000
        
    
        #14
        b["dataset"]["test_dir"]="../datasets/CIFAR100/test/"+class000
        
        
    
        #13
        b["dataset"]["val_dir"]="../datasets/CIFAR100/test/"+class000
        
        
        
    
        #change k 28
        b["hparams"]["k"]=k000
      
    
        #change_per_k 29
        b["hparams"]["per_k"]=per_k000
        
    
        

        name=b["dataset"]["train_dir"].split("/")[-1]
        #print(name)    
        k_name=b["hparams"]["k"]
        #print(k_name)
        per_k_name= b["hparams"]["per_k"]
        #print(per_k_name)

        #change save dir 18
        b["logger"]["save_dir"]= "./logs_"+"__class="+name+"__k="+str(k_name)+"__per_k="+str(per_k_name)
        
        b["hparams"]["inverse"]=resnet_iden
       
        if b["hparams"]["inverse"]==0:
            resnet_name="__resnet"
        elif b["hparams"]["inverse"]==1:
             resnet_name="__inverseresnet"
        #print(resnet_name)
        config_name_with_specific_or_general__k_and_per_k="./configs/config"+"__class="+name+"__k="+str(k_name)\
        +"__per_k="+str(per_k_name)+resnet_name+".yaml"
        #print(config_name_with_specific_or_general__k_and_per_k)
        
        b["logger"]["save_dir"]="./logs/"+"class="+name+"__k="+str(k_name)\
        +"__per_k="+str(per_k_name)+resnet_name
    
         
    with open(config_name_with_specific_or_general__k_and_per_k,"w") as new:
        yaml.dump(b,new)
        new.close()

        
    return config_name_with_specific_or_general__k_and_per_k




def check_config(config_name_with_specific_or_general__k_and_per_k):#use to check if the config generate well, if it has right name
    a=config_name_with_specific_or_general__k_and_per_k
    a=a.split("=")
    class_type=a[1].split("__")[0]
    
    k123=a[2].split("__")[0]
    
    per_k123=a[3].split("__")[0]
    
    resnet_type=a[3].split("__")[1].split(".")[0]
    
    with open(config_name_with_specific_or_general__k_and_per_k,"r") as new:
        check_config_tem=yaml.safe_load(new)
        #print("check train dataset:", class_type in check_config_tem["dataset"]["train_dir"])
        #print("check test dataset:", class_type in check_config_tem["dataset"]["test_dir"])
        #print("check val dataset:", class_type in check_config_tem["dataset"]["val_dir"])
        
        #print("check k:",float(k123)==check_config_tem["hparams"]["k"])
        #print("check per_k:",float(per_k123)==check_config_tem["hparams"]["per_k"])
        #if check_config_tem["hparams"]["inverse"] ==0:
            #print("resnet type:",resnet_type=="resnet")
        #else:
            #print("resnet type:",resnet_type=="inverseresnet")
        
        #print("\n")
        
        if class_type in check_config_tem["dataset"]["train_dir"] and class_type in check_config_tem["dataset"]["test_dir"]\
        and class_type in check_config_tem["dataset"]["val_dir"] and float(k123)==check_config_tem["hparams"]["k"] \
        and float(per_k123)==check_config_tem["hparams"]["per_k"] and check_config_tem["hparams"]["inverse"] ==0\
        and resnet_type=="resnet":
            print("good!!!!!!!!!!!!")
            
        elif   class_type in check_config_tem["dataset"]["train_dir"] and class_type in check_config_tem["dataset"]["test_dir"]\
        and class_type in check_config_tem["dataset"]["val_dir"] and float(k123)==check_config_tem["hparams"]["k"] \
        and float(per_k123)==check_config_tem["hparams"]["per_k"] and check_config_tem["hparams"]["inverse"] ==1\
        and resnet_type=="inverseresnet":
            print("good!!!!!!!!!!!!") 
             
        
        else:
            print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            
        print("\n")



