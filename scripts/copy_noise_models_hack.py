from sigvisa import Sigvisa

def main():
    input_runid=25
    output_runid=27
    phase='noise_raw'
    
    s = Sigvisa()
    query = """
CREATE TEMPORARY TABLE spm2 ENGINE=MEMORY SELECT * FROM sigvisa_param_model WHERE fitting_runid=%d and phase='%s';
UPDATE spm2 SET fitting_runid=%d;
UPDATE spm2 SET modelid=NULL;
INSERT INTO sigvisa_param_model SELECT * FROM spm2;
DROP TABLE spm2;""" % (input_runid, phase, output_runid)
    print query
    print "not executing because this account probably doesn't have db write permission, but this query should work to run as root."
    
    #cursor = s.dbconn.cursor()
    #cursor.execute(query)
    #cursor.close()
    #s.dbconn.commit()
    

if __name__=="__main__":
    main()

