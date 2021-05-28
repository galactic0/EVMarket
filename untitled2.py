# -*- coding: utf-8 -*-
"""
Created on Mon May 24 13:53:00 2021

@author: Aman
"""

import urllib3



http = urllib3.PoolManager()
re=http.request("GET","https://play.google.com/store/apps/details?id=com.metrobikes.app&showAllReviews=true")
