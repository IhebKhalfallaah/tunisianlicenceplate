FROM mysql:8.0

ENV MYSQL_DATABASE=license_plate
ENV MYSQL_USER=back
ENV MYSQL_PASSWORD=password
ENV MYSQL_ROOT_PASSWORD=rootpassword
COPY schema.sql docker-entrypoint-initdb.d/init.sql
EXPOSE 3306