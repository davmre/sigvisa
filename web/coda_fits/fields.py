from django.db import models
from datetime import datetime
import time
import dateutil.parser

class UnixTimestampField(models.DateTimeField):
    """UnixTimestampField: creates a DateTimeField that is represented on the
    database as a TIMESTAMP field rather than the usual DATETIME field.
    """

    __metaclass__ = models.SubfieldBase
#    def __init__(self, null=False, blank=False, **kwargs):
#        super(UnixTimestampField, self).__init__(**kwargs)

    def db_type(self, connection):
        return "FLOAT"

    def to_python(self, value):
        if isinstance(value, unicode):
            a  = dateutil.parser.parse(value)
        else:
            a = datetime.fromtimestamp(value)
        print "given", value, "returning", a
        return a

    def get_db_prep_value(self, value, connection, prepared=False):
        if not prepared:
            value = self.get_prep_value(value)
        if value==None:
            return None
        b= time.mktime(value.timetuple())
        print "prep given", value, "returning", b
        return b
