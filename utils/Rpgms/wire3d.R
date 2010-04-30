require("lattice")

### Basic Functions

	### Create rounded values
	binBuilder<-function(x,s){
		bigger<-x*(1/s)
		bigger<-round(bigger)
		return(bigger*s)
	}

	### Colors for level plot
	myspectr<-function(numCols){c(rev(hsv(4/6,seq(0,numCols)/numCols,1)),hsv(0,seq(0,numCols)/numCols,1))}

	### Unfactorize into numeric
	f2n<-function(x){as.numeric(as.character(x))}

### Data Manipulation Function

wireDF<-function(x,main_label,mb_bins=.25,d_bins=10,wire=TRUE,probs=TRUE){
	#Require x to be a dataframe with fields: dist,mag,det

	x$mag<-binBuilder(x$mag,mb_bins)
	x$dist<-binBuilder(x$dist,d_bins)

	tot_evs<-dim(x)[1]
	tot_dets<-sum(x$det)
	### Aggregated across array and station
	
	all_ev<-as.data.frame(table(x$mag,x$dist))
	det_ev<-as.data.frame(table(x$mag[x$det==1],x$dist[x$det==1]))

	all_ev$det<-0
	for(i in seq(1,dim(all_ev)[1])){
		j<-f2n(all_ev[i,1])
		k<-f2n(all_ev[i,2])
	
		w<-f2n(det_ev[,1])==j & f2n(det_ev[,2])==k
		if(sum(w)){all_ev$det[i]<-det_ev[w,3]}
	}

	all_cnts<-all_ev


	colnames(all_cnts)<-c("mb","dist","cnt","det")
	all_cnts$prob<-all_cnts$det/all_cnts$cnt
	all_cnts$prob[is.na(all_cnts$prob)]<-0

	
	### Plots

        if(probs){
		if(wire){
			print(wireframe(prob~mb*dist,data=all_cnts,xlab="Magnitude",ylab="Distance",
				zlab="Probability",main=paste(main_label," Probability of Detection: ",tot_dets,"/",tot_evs)))
		}else{
	
			print(levelplot(prob~dist*mb,data=all_cnts,cuts=100,
				labels=FALSE,pretty=TRUE,ylab="Magnitude",scale=list(x=list(rot=45)),
				main=paste(main_label," Probability of Detection: ",tot_dets,"/",tot_evs),
				xlab="Distance",col.regions=(myspectr(400))))}
	}else{
		print(levelplot(cnt~dist*mb,data=all_cnts,cuts=100,
			labels=FALSE,pretty=TRUE,ylab="Magnitude",scale=list(x=list(rot=45)),
			main=paste(main_label," Counts: ",tot_dets,"/",tot_evs),
			xlab="Distance",col.regions=(myspectr(400))))
	}

}

wireDF_dep<-function(x,main_label,dep_bins=25,d_bins=10){
	#Require x to be a dataframe with fields: dist,dep,det

	x$dep<-binBuilder(x$dep,dep_bins)
	x$dist<-binBuilder(x$dist,d_bins)

	tot_evs<-dim(x)[1]
	tot_dets<-sum(x$det)
	### Aggregated across array and station
	
	all_ev<-as.data.frame(table(x$dep,x$dist))
	det_ev<-as.data.frame(table(x$dep[x$det==1],x$dist[x$det==1]))

	all_ev$det<-0
	for(i in seq(1,dim(all_ev)[1])){
		j<-f2n(all_ev[i,1])
		k<-f2n(all_ev[i,2])
	
		w<-f2n(det_ev[,1])==j & f2n(det_ev[,2])==k
		if(sum(w)){all_ev$det[i]<-det_ev[w,3]}
	}

	all_cnts<-all_ev


	colnames(all_cnts)<-c("dep","dist","cnt","det")
	all_cnts$prob<-all_cnts$det/all_cnts$cnt
	all_cnts$prob[is.na(all_cnts$prob)]<-0

	
	### Plots
	
	print(levelplot(prob~dist*dep,data=all_cnts,cuts=100,
		labels=FALSE,pretty=TRUE,ylab="Depth",scale=list(x=list(rot=45)),
		main=paste(main_label," Probability of Detection: ",tot_dets,"/",tot_evs),
		xlab="Distance",col.regions=(myspectr(400))))

}




# Output probability tables for each of the phases for dist and magnitude

prob_table<-function(x,main_label,mb_bins=.25,d_bins=5){
	#Require x to be a dataframe with fields: dist,mag,det

	x$mag<-binBuilder(x$mag,mb_bins)
	x$dist<-binBuilder(x$dist,d_bins)

	tot_evs<-dim(x)[1]
	tot_dets<-sum(x$det)
	### Aggregated across array and station
	
	all_ev<-as.data.frame(table(x$mag,x$dist))
	det_ev<-as.data.frame(table(x$mag[x$det==1],x$dist[x$det==1]))

	all_ev$det<-0
	for(i in seq(1,dim(all_ev)[1])){
		j<-f2n(all_ev[i,1])
		k<-f2n(all_ev[i,2])
	
		w<-f2n(det_ev[,1])==j & f2n(det_ev[,2])==k
		if(sum(w)){all_ev$det[i]<-det_ev[w,3]}
	}

	all_cnts<-all_ev


	colnames(all_cnts)<-c("mb","dist","cnt","det")
	all_cnts$prob<-all_cnts$det/all_cnts$cnt
	all_cnts$prob[is.na(all_cnts$prob)]<-0
        all_cnts$phase<-main_label

	return(all_cnts)
}

# Output probability tables for each of the phases for depth, distance, and magnitude

probMddTable<-function(x,mb_bins=.75,d_bins=12,dep_bins=300){
	#Require x to be a dataframe with fields: dist,mag,det

	x$mag<-binBuilder(x$mag,mb_bins)
	x$dist<-binBuilder(x$dist,d_bins)
	x$dep<-binBuilder(x$dep,dep_bins)

	tot_evs<-dim(x)[1]
	tot_dets<-sum(x$det)
	### Aggregated across array and station
	
	all_ev<-as.data.frame(table(x$mag,x$dep,x$dist))
	det_ev<-as.data.frame(table(x$mag[x$det==1],x$dep[x$det==1],x$dist[x$det==1]))
     

	all_ev$det<-0
       
	for(i in seq(1,dim(all_ev)[1])){
	       
		j<-f2n(all_ev[i,1])
		k<-f2n(all_ev[i,2])
                l<-f2n(all_ev[i,3])
	
		w<-f2n(det_ev[,1])==j & f2n(det_ev[,2])==k & f2n(det_ev[,3])==l
		if(sum(w)){all_ev$det[i]<-det_ev[w,4]}
	}

	all_cnts<-all_ev
       

	colnames(all_cnts)<-c("mag","dep","dist","cnt","det")
        
	all_cnts$prob<-all_cnts$det/all_cnts$cnt
	all_cnts$prob[is.na(all_cnts$prob)]<-0
        all_cnts<-all_cnts[,-c(4,5)]
        all_cnts<-all_cnts[,c(4,1,2,3)]
	return(all_cnts)
}


