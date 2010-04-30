## Logistic regression requires transformation of distance and magnitude for
## prediction.  This varies by phase.  Function, featBuilder, creates these 
## features.  
## Paul Kidwell 04-26-2010

## Transform a vector in terms of a specific basis function

gtf<-function(y,feat,mu,sig){
    temp<-y[,colnames(y)==feat]
    temp<-dnorm(temp,mean=mu,sd=sig)
    return(temp)
}

## Delete variables of a given name from a dataframe

delVar<-function(x,vars){
   ind<-seq(1,dim(x)[2])[colnames(x) %in% vars]
   x<-x[,-ind]
   return(x)
}

## Create data frame with correct variables
featBuilder<-function(x,phase){

    if(phase=="P"){
        #x$dist<-gtf(x,"dist",60,200)
        #x$md<-(7-x$mag)*x$dist
        #x$dist0<-gtf(x,"dist",0,2)
        #x$mag275<-gtf(x,"mag",2.75,.2)
        # No features needed...

    }else if(phase=="S"){

        x$dist0<-gtf(x,"dist",0,5)
        x$md<-(7-x$mag)*x$dist

    }else if(phase=="PcP"){  ##Good
        x$dist40<-gtf(x,"dist",40,20)
        x$mag6<-gtf(x,"mag",6,5.5)
        x<-delVar(x,c("dist","mag"))
	
    }else if(phase=="pP"){  ## Return if time

    }else if(phase=="ScP"){  ##Good
        x$dist35<-gtf(x,"dist",35,20)
        x$mag6<-gtf(x,"mag",6,5.5)
        x<-delVar(x,c("dist","mag"))

    }else if(phase=="PKKPbc"){  ##Good
        x$dist12540<-gtf(x,"dist",125,40)
        x$mag68<-gtf(x,"mag",6,8)
        x<-delVar(x,c("dist","mag"))
	
    }else if(phase=="PKP"){   ##Good
          x$dist170<-gtf(x,"dist",170,20)
          x$dist12520<-gtf(x,"dist",125,20)
          x$md<-(7-x$mag)*x$dist
          x<-delVar(x,c("dist"))

    }else if(phase=="Sn"){  ##Good
         x$md<-(7-x$mag)*x$dist

    }else if(phase=="Lg"){  ##Good
	 x<-delVar(x,c("dep"))

    }else if(phase=="Pn"){  ##Good
       
    }else if(phase=="Pg"){  ##Good
        x<-delVar(x,c("dep"))

    }else if(phase=="Rg"){  ##Good
        x<-delVar(x,c("dep"))

    }else if(phase=="PKPbc"){  ##Good
        x$dist145<-gtf(x,"dist",145,10)
        #x$mag6<-gtf(x,"mag",6,5.5)
        #x$md145<-(7-x$mag)*x$dist145
        x<-delVar(x,c("dist"))
	
    }else if(phase=="PKPab"){  ##Good
        x$dist175<-gtf(x,"dist",175,30)
        x$mag6<-gtf(x,"mag",6,5.5)
        x<-delVar(x,c("dist","mag"))
	
    }else{print("Phase not found")}

   return(x)
}




### Build a matrix of all combinations of magnitude, depth, and distance
### used when adding non-detections for logistic regression models

gridPoints<-function(mag,dep,dist){
   a<-length(mag)
   b<-length(dep)
   d<-length(dist)

   sample_pts<-matrix(0,a*b*d,4)
   n<-1

   for(i in 1:a){
       for(j in 1:b){
           for(k in 1:d){
               sample_pts[n,2:4]<-c(mag[i],dep[j],dist[k])
               n<-n+1
           }
       }
   }

   sample_pts<-data.frame(sample_pts)

   colnames(sample_pts)<-c("det","mag","dep","dist")
   return(sample_pts)
}


### Adds missed detections at when appropriate due geophysic impossibility

#addNonDetects<-function(x,phase){
#
#    if(phase=="PKP"){   
#       x<-rbind(x,gridPoints(seq(2,6,.1),0,c(seq(0,110),seq(130,180))))
#  
#    }else if(phase=="PKPbc"){  
#       x<-rbind(x,gridPoints(seq(2,6,.1),0,c(seq(0,140),seq(160,180))))
#
#    }else{print("Phase not found")}
#
#   return(x)
#}


### Create a dataframe with all variables being used in regression
### Input: a dataframe with mag,dep,dist

dfAllVars<-function(x){

    x$dist0<-gtf(x,"dist",0,5)
    x$dist35<-gtf(x,"dist",35,20)
    x$dist40<-gtf(x,"dist",40,20)
    x$dist12520<-gtf(x,"dist",125,20)
    x$dist12540<-gtf(x,"dist",125,40)
    x$dist145<-gtf(x,"dist",145,10)
    x$dist170<-gtf(x,"dist",170,20)
    x$dist175<-gtf(x,"dist",175,30)

    x$mag6<-gtf(x,"mag",6,5.5)
    x$mag68<-gtf(x,"mag",6,8)
     
    x$md<-(7-x$mag)*x$dist
    x$md145<-(7-x$mag)*x$dist145

    return(x)
} 
   

### Format data, create an additional column containing the phase label
### for each event

dfFormat<-function(dets){
    phasenames<-colnames(dets)[-seq(1,5)]
    phaseid<-dets[,-seq(1,5)]
    dets$phase<-apply(phaseid,1,function(x){phasenames[x==1]})
    return(dets)
}   

       

