.. _sysID:

System Identification (WIP)
============================

This section provides a comprehensive guide to perform system identification (SysID) for Dynamixel motors.

Hardware Setup
---------------------------------------------
You will need to purchase the sysID section in the :ref:`bill_of_materials`.

For 3D-printed parts, you can find all the sliced plates in the `MakerWorld <https://makerworld.com/en/models/1068768>`_ 
and the CAD files in the `Onshape document <https://cad.onshape.com/documents/1370cb70ae00945ee5a1ab36>`_.

We use 21700 cells as loads to adjust the motor load weight.

TODO: Add the image of the sysID setup.

SysID Data Collection
---------------------------------------------
TODO

Set up Optuna Dashboard
---------------------------------------------

To visualize the sysID optimization process, you need to install the following packages:

.. tabs::

   .. group-tab:: Linux

      ::

         sudo apt install libpq-dev postgresql
         sudo systemctl start postgresql

   .. group-tab:: Mac OSX (arm64)

      ::

         brew install postgresql
         brew services start postgresql

Run PostgreSQL:

.. tabs::

   .. group-tab:: Linux

      ::

         sudo -u postgres psql

   .. group-tab:: Mac OSX (arm64)

      ::

         psql postgres

Enter the following commands in the PostgreSQL prompt:

::

   CREATE DATABASE optuna_db;
   CREATE USER optuna_user WITH ENCRYPTED PASSWORD 'password';
   GRANT ALL PRIVILEGES ON DATABASE optuna_db TO optuna_user;

Exit the PostgreSQL prompt.

Run the Optuna dashboard:

::

   optuna-dashboard postgresql://optuna_user:password@localhost/optuna_db