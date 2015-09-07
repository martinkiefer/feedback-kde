#Get nlopt
wget http://ab-initio.mit.edu/nlopt/nlopt-2.4.2.tar.gz
tar -xvzf nlopt-2.4.2.tar.gz

#Compile and install nlopt
cd nlopt-2.4.2
patch -p0 < nlopt-patch.diff
./configure
make clean
make
su -c make install
cd ..

#Compile and install postgres
./configure -with-opencl
make clean
make
su -c make install

#Setup
/usr/local/pgsql/bin/initdb ./experiments
/usr/local/pgsql/bin/pg_ctl -D ./experiments -l logfile start
sleep 3
/usr/local/pgsql/bin/dropdb experiments
/usr/local/pgsql/bin/createdb experiments
bash ./analysis/static/datasets/prepare.sh
bash ./analysis/static/timing/prepare_experiment.sh
cp ./analysis/conf.sh.template ./analysis/conf.sh
/usr/local/pgsql/bin/pg_ctl -D ./experiments -l logfile stop

#Insert sanity check here.
