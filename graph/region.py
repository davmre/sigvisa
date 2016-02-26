from sigvisa import Sigvisa

class Region(object):

    def __init__(self, lons, lats, times, rate_bulletin="leb", 
                 min_mb=None,
                 rate_train_start=1238889600, rate_train_end=1245456000):
        self.left_lon = lons[0] if lons is not None else None
        self.right_lon = lons[1] if lons is not None else None
        self.bottom_lat = lats[0] if lats is not None else None
        self.top_lat = lats[1] if lats is not None else None
        self.stime = times[0] if times is not None else None
        self.etime = times[1] if times is not None else None

        self.rate_train_start = 1238889600
        self.rate_train_end = 1245456000
        self.event_rate = self._estimate_event_rate(rate_train_start, rate_train_end, 
                                                    min_mb=min_mb,
                                                    bulletin=rate_bulletin)

    def contains_event(self, ev=None, lon=None, lat=None, time=None):
        if ev is not None:
            lon, lat, time = ev.lon, ev.lat, ev.time
        if self.left_lon is not None and lon is not None:
            if self.left_lon < self.right_lon:
                if lon < self.left_lon or lon > self.right_lon:
                    return False
            else:
                # if the region wraps around the date line,
                if lon < self.left_lon and ev > self.right_lon:
                    return False
        if self.bottom_lat is not None and lat is not None:
            if self.bottom_lat > lat or self.top_lat < lat:
                return False
        if self.stime is not None and time is not None:
            if self.stime > time or self.etime < time:
                return False
        return True

    def area_deg(self):
        width = (self.right_lon - self.left_lon) % 360
        height = self.top_lat - self.bottom_lat
        return width*height

    def _estimate_event_rate(self, train_start, train_end, min_mb=None, bulletin="leb"):
        mb_cond = "and mb < %f" %min_mb if min_mb is not None else ""
        sql_query = "select evid from %s_origin where (lon between %.1f and %.1f) and (lat between %.1f and %.1f) and (time between %.1f and %.1f) %s" % (bulletin, self.left_lon, self.right_lon, self.bottom_lat, self.top_lat, train_start, train_end, mb_cond)
        s = Sigvisa()
        cursor = s.dbconn.cursor()
        cursor.execute(sql_query)
        results=  cursor.fetchall()
        cursor.close()
        n_events = len(results)
        dtime = train_end - train_start
        rate = n_events / float(dtime)
        return rate
