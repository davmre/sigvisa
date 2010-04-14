import numpy as np

AVG_EARTH_RADIUS_KM = 6371.0

# strip away any trailing errors which can cause arccos to return nan
# if mat is -1.0000000000000002 for example
def safe_acos(mat):
  return np.where(mat > 1, 0., np.where(mat < -1, np.pi, np.arccos(mat)))

def dist_deg(loc1, loc2):
  """
  Compute the great circle distance between two point on the earth's surface
  in degrees.
  loc1 and loc2 are pairs of longitude and latitude
  >>> int(dist_deg((10,0), (20, 0)))
  10
  >>> int(dist_deg((10,0), (10, 45)))
  45
  >>> int(dist_deg((-78, -12), (-10.25, 52)))
  86
  >>> dist_deg((132.86521, -0.45606493), (132.86521, -0.45606493)) < 1e-4
  True
  >>> dist_deg((127.20443, 2.8123965), (127.20443, 2.8123965)) < 1e-4
  True
  """
  lon1, lat1 = loc1
  lon2, lat2 = loc2

  return np.degrees(safe_acos(np.sin(np.radians(lat1))
                              * np.sin(np.radians(lat2))
                              + np.cos(np.radians(lat1))
                              * np.cos(np.radians(lat2))
                              * np.cos(np.radians(lon2 - lon1))))

def dist_km(loc1, loc2):
  """
  Returns the distance in km between two locations specified in degrees
  loc = (longitude, latitude)
  """
  lon1, lat1 = loc1
  lon2, lat2 = loc2
  
  return np.radians(dist_deg(loc1, loc2)) * AVG_EARTH_RADIUS_KM

def degdiff(angle1, angle2):
  """
  The difference of two angles given in degrees. The answer is an angle from
  -180 to 180. Positive angles imply angle2 is clockwise from angle1 and -ve
  angles imply counter-clockwise.
  
  >>> int(degdiff(40, 30))
  -10
  >>> int(degdiff(30, 40))
  10
  >>> int(degdiff(361, 40))
  39
  >>> int(degdiff(40, 361))
  -39
  >>> degdiff(40,250)
  -150
  >>> degdiff(40,200)
  160
  >>> degdiff(40, 219)
  179
  >>> degdiff(40, 220)
  180
  >>> degdiff(40, 221)
  -179
  """
  # bring the angle into the 0 to 360 range
  delta = ((angle2 - angle1) + 360) % 360
  # angles above 180 need to be shifted down by 360 degrees so that 181 is -179
  # 200 is -160 etc.
  return delta - (delta > 180) * 360

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
