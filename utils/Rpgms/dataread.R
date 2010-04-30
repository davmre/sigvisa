## Read in data from a text file written from Python learn.py

loadfile<-function(filein){
    y<-read.table(file=filein,header=TRUE)

    phasenames<-colnames(y)[-seq(1,5)]
    phaseid<-y[,-seq(1,5)]
    y$phase<-apply(phaseid,1,function(x){phasenames[x==1]})
    return(y)
}


