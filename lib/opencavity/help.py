# -*- coding: utf-8 -*-
'''
Created on 15 feb. 2015

@author: mohamed seghilani
'''
import opencavity
import webbrowser
import platform

#if __name__ == '__main__':
    
def launch():
    help_path=opencavity.__file__
    if platform.system()=='Windows':
        separator='\\'
    else:
        separator='/'
        
    count=1
    while (not help_path.endswith(separator)) and count<50:
        help_path=help_path[:-1]
        count=count+1 #prevent unfinit loop if path is empty
    help_path2='Docs/_build/html/index.html'
    help_path=help_path+help_path2
    webbrowser.open(help_path)
