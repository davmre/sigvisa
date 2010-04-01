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

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
