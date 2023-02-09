import argparse

parser = argparse.ArgumentParser(description='RayTracing for sun glitter vis')

# arguments
parser.add_argument('--solar_altitude', type=float, default=45, 
                    help='solar_altitude (angle in deg from the horizontal)')
parser.add_argument('--solar_azimuth', type=float, default=0,
                    help='solar_azimuth (angle in deg from the i axis (along wind direction))')
parser.add_argument('--wind_speed', type=float, default=1,
                    help='wind speed in m/s')
parser.add_argument('--n', type=int, default=7,
                    help='order of hexagonal domain')
parser.add_argument('--iter', type=int, default=5000,
                    help='number of parent rays to seed')
parser.add_argument('--save_fp',type=str, 
                    help='directory on where to save the daughter rays')
parser.add_argument('--camera_altitude', type=float, default=45, 
                    help='camera_altitude (angle in deg from the horizontal)')
parser.add_argument('--camera_azimuth', type=float, default=180,
                    help='camera_azimuth (angle in deg from the i axis (along wind direction))')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()
