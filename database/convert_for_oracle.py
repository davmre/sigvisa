import os, sys, traceback, pdb, re

from optparse import OptionParser

def convert_to_oracle(text):
    """
    
    Convert an SQL file from MySQL to Oracle format. At the moment this includes:
         - Replicate MySQL auto_increment with an explicit sequence/trigger combo.

    Input:
    text: string containing the text of an SQL file

    Returns: text of the converted file.

    """

    lines = text.splitlines()

    create_table_re = re.compile(r"create table (\w+)\s*\(")
    auto_increment_re = re.compile(r"(\s*)(\w+)(\s+)(.+)auto_increment,")

    new_lines = []

    auto_incr_vars = []

    for line in lines:
        m = create_table_re.search(line)
        if m is not None:
            current_table = m.group(1)
            new_lines.append(line)
            continue

        m = auto_increment_re.search(line)
        if m is not None:
            indent = m.group(1)
            varname = m.group(2)
            padding = m.group(3)
            attributes = m.group(4)
            auto_incr_vars.append((varname, current_table))
            new_lines.append("%s%s%s%s," % (indent, varname, padding, attributes))
        else:
            new_lines.append(line)
            
    for (varname, table) in auto_incr_vars:
        seq_name = "%s_seq" % varname
        trigger_name = "%s_trigger" % varname
        l = ["",
             "create sequence %s start with 1 increment by 1 nomaxvalue;" % seq_name,
             "create or replace trigger %s" % trigger_name,
             "before insert on %s" % table,
             "for each row",
             "begin",
             "select %s.nextval into :new.%s from dual;" % (seq_name, varname),
             "end;",
             "/",
             "",]
        new_lines += l
    return "\n".join(new_lines)


def main():

    parser = OptionParser()
    #parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str", help="")
    (options, args) = parser.parse_args()

    for fname in args:
        with open(fname, 'r') as f_read:
            text = f_read.read()

        new_fname = os.path.splitext(fname)[0] + "_oracle.sql"
        print "converting %s to oracle SQL, saving to %s ..." % (fname, new_fname),
        new_text = convert_to_oracle(text)
        with open(new_fname, 'w') as f_write:
            f_write.write(new_text)
        print "done"

if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
