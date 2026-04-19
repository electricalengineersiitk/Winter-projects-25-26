import numpy as np
from formation import uav_group

MAP_LIMIT = 100
MISSION_START = (5, 15)
MISSION_GOAL = (100, 100)

HAZARD_CENTER = (50, 50)
HAZARD_WIDTH = 24

CORE_L = HAZARD_CENTER[0] - (HAZARD_WIDTH / 2)
CORE_R = HAZARD_CENTER[0] + (HAZARD_WIDTH / 2)
CORE_B = HAZARD_CENTER[1] - (HAZARD_WIDTH / 2)
CORE_T = HAZARD_CENTER[1] + (HAZARD_WIDTH / 2)

buffer_margin = 1.5
PLAN_L = CORE_L - uav_group.radius_buffer - buffer_margin
PLAN_R = CORE_R + uav_group.radius_buffer + buffer_margin
PLAN_B = CORE_B - uav_group.radius_buffer - buffer_margin
PLAN_T = CORE_T + uav_group.radius_buffer + buffer_margin