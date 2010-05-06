### Functions for logistic regression fitting and prediction
### Paul Kidwell 4-23-2010

## Regression function
logReg<-function(train,test){
    # Input a test and training data set with the same layout
    # where 1st column is response
    # Output: returns a list of coefs, preds, avg likelihood

    betas<-logReg.fit(train)
    preds<-logReg.pred(test,betas)
    like<-logReg.like(test[,1],preds)
    return(list(betas,preds,like))

}


## Fit Regression
logReg.fit<-function(train){
    # Input training data as a dataframe with the response in the 
    # first column.  Return the coefficients.  

    terms<-paste(colnames(train[-1]),collapse="+")
    eqn<-paste(colnames(train[1]),terms,sep="~")

    m1<-glm(eqn,data=train,family=binomial)
    print(summary(m1))

    return(m1$coefficients)
}


## Predictions
logReg.pred<-function(newdata,coefs){
    #Input a dataframe with the data to predict and a 
    # vector of regression coefficients
    if(dim(newdata)[1]>0){
        prednew <- coefs[1] + (as.matrix(newdata[,-1]) %*% coefs[-1])
        preds<-exp(prednew)/(1+exp(prednew))
    }else{preds<-NaN}

    return(preds)
}
	


### Average likelihood evaluation
logReg.like<-function(actual,det_prob){
    ## Input the actual binary observation and the probability of 
    ## a detection

    not_det_prob<-1-det_prob

    not_probs<-log(not_det_prob[actual==0])
    det_probs<-log(det_prob[actual==1])	
	
    avelikeli<-mean(c(not_probs,det_probs))
	
    return(avelikeli)
}


### Average log likelihood over a number of different models 
wgtlike<-function(loglike,cnts){
    ## A weighted average of loglikelihood based upon the number of training
    ## examples for each model.  Input a vector of likelihoods and number of test samples

    return(sum(loglike*cnts,na.rm=TRUE)/sum(cnts[!is.na(loglike)]))
}


###

### Logistic surface for specific sites


heatPredSurfSite<-function(main_label,betas){
    require("lattice")
    source("functions/dfconversion.R")    

    myspectr<-function(numCols){c(rev(hsv(4/6,seq(0,numCols)/numCols,1)),hsv(0,seq(0,numCols)/numCols,1))}


    ## create data frame of grid values
    dists<-seq(1,180)
    mags<-seq(2,6,.1)
    fig_in<-as.data.frame(matrix(0,length(dists)*length(mags),4))
    colnames(fig_in)<-c("det","mag","dep","dist")
    fig_in$mag<-sort(rep(mags,length(dists)))
    fig_in$dist<-rep(dists,length(mags))
    fig_old<-fig_in   

    fig_in<-featBuilder(fig_in,main_label)
    fig_in$probs<-logReg.pred(fig_in,betas)
    fig_in<-cbind(fig_in$probs,fig_old)
    colnames(fig_in)[1]<-"probs"
    
   
    

    ## output a level plot

    print(levelplot(probs~dist*mag,data=fig_in,cuts=100,
	labels=FALSE,ylab="Magnitude",main=main_label,
	xlab="Distance",col.regions=(myspectr(400))))

    #print(wireframe(probs~dist*mag,data=fig_in,xlab="Distance",ylab="Magnitude",
	#zlim=c(0,1),zlab="Probability",main=main_label))

}



### Logistic surface


heatPredSurf<-function(main_label,betas){
    require("lattice")
    source("functions/dfconversion.R")    

    myspectr<-function(numCols){c(rev(hsv(4/6,seq(0,numCols)/numCols,1)),hsv(0,seq(0,numCols)/numCols,1))}


    ## create data frame of grid values
    dists<-seq(1,180)
    mags<-seq(2,6,.1)
    fig_in<-as.data.frame(matrix(0,length(dists)*length(mags),4))
    colnames(fig_in)<-c("det","mag","dep","dist")
    fig_in$mag<-sort(rep(mags,length(dists)))
    fig_in$dist<-rep(dists,length(mags))
    fig_old<-fig_in   

    fig_in<-featBuilder(fig_in,main_label)
    fig_in$probs<-logReg.pred(fig_in,betas)
    fig_in<-cbind(fig_in$probs,fig_old)
    colnames(fig_in)[1]<-"probs"
    print(head(fig_in))
    

    ## output a level plot
    postscript(file=paste(main_label,".heatmap.ps",sep=""))

    print(levelplot(probs~dist*mag,data=fig_in,cuts=100,
	labels=FALSE,ylab="Magnitude",main=main_label,
	xlab="Distance",col.regions=(myspectr(400))))

    print(wireframe(probs~dist*mag,data=fig_in,xlab="Distance",ylab="Magnitude",
	zlim=c(0,1),zlab="Probability",main=main_label))

    dev.off()

    return(fig_in)
}



### Dataframe for outputs of all logistic regressions.  
### Includes all features used in all regressions

coefDF<-function(n){
    vars<-c("phase","site","(Intercept)","mag","dep","dist","dist0","dist35","dist40",
        "dist12520","dist12540","dist145","dist170","dist175",
        "mag6","mag68","md")
    x<-data.frame(matrix(0,n,length(vars)))
    colnames(x)<-vars
    return(x)
} 
