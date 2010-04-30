### Perform empirical bayes estimate of site specific logistic regression models
### The dfConversion file creates the appropriate feature vector
### Output: a model for each phase*site combination...
### Input: training data, testing data, dfConversion file,
### Output: features estimates file (matrix), parameters estimate file (list), 
###       diagnostics (likelihood, predictive surface)
### Paul Kidwell 4-23-2010


FitLogistic<-function(infile,outfile){

   pwd<-system("pwd",intern=TRUE)
   source(paste(pwd,"/utils/Rpgms/dataread.R",sep=""))
   source(paste(pwd,"/utils/Rpgms/dfconversion.R",sep=""))
   source(paste(pwd,"/utils/Rpgms/logistic.R",sep=""))
   source(paste(pwd,"/utils/Rpgms/wire3d.R",sep=""))
   

   
   dets<-loadfile(infile)

   ## Select variables and create constants and output vectors

   remove_cols<-c(1,seq(6,20))
   phasenames<-unique(dets$phase)
   sites<-unique(dets$site)
   log_coef<-coefDF(length(sites)*length(phasenames))


   ## Logistic regression for each phase
  
   w<-1
   for(j in phasenames){
    
       ### Create phase specific dataset
       train<-dets[dets$phase==j,-remove_cols]
       train_site<-dets$site[dets$phase==j]
   
       ### Tabulate emprical distribution over magnitude and distance, and create pseudo-data
       prior<-probMddTable(train,mb_bins=.5,d_bins=10,dep_bins=300)
       det<-1*(runif(dim(prior)[1])<=prior[,1]) 
       pseudodata<-as.data.frame(apply(cbind(det,prior[,-1]),2,function(x){as.numeric(x)}))

       ### Build features
       train<-featBuilder(train,j)
       pseudodata<-featBuilder(pseudodata,j)
    

       ### Loop overall sites and build a model, output coefficients, site specific plots
       for(s in sites){
           print(s)
           train_site_df<-as.data.frame(apply(rbind(train[train_site==s,],
               pseudodata),2,function(x){as.numeric(x)}))
       
           out<-list(logReg.fit(train_site_df))
         
           log_coef$phase[w]<-j
           log_coef$site[w]<-s
           for(i in names(out[[1]])){
	       log_coef[w,colnames(log_coef)==i]<-out[[1]][names(out[[1]])==i]}
           w<-w+1
        }
 
 
    }

    write.csv(log_coef,file=outfile,row.names=FALSE)
    return()
}


#FitLogistic("EventDetectionAnalysis.txt",outfile="EventDetectionPriorEstimates.csv")
