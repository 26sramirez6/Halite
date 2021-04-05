rm -rf halite_swig_wrap.o _halite_swig.so halite_swig_wrap.cxx
swig -c++ -python halite_swig.i
/usr/local/bin/c++ -fPIC -I/home/saul.ramirez/anaconda3/include/python3.7m -I/home/saul.ramirez/eclipse-workspace/Halite -c halite_swig_wrap.cxx
/usr/local/bin/c++ -fPIC -shared halite_swig_wrap.o -o _halite_swig.so
