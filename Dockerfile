FROM continuumio/miniconda3

# Install git
RUN apt-get update && apt-get install -y git

# Créer et activer un nouvel environnement conda
RUN conda create -n dash_app python=3.9 -y
SHELL ["conda", "run", "-n", "dash_app", "/bin/bash", "-c"]

# RUN git clone https://github.com/dcarrasquet/iMDiNE.git /app
# RUN ls -l /app
# Copier le fichier des dépendances (environment.yml)

COPY environment.yml /tmp/environment.yml

# RUN mkdir -p /app
# COPY results /app/results
# COPY scripts /app/scripts

# Installer les dépendances
RUN conda env update -n dash_app -f /tmp/environment.yml

# Activer l'environnement conda
RUN echo "conda activate dash_app" >> ~/.bashrc
SHELL ["conda", "run", "-n", "dash_app", "/bin/bash", "-c"]

# Exposer le port sur lequel Dash va tourner
EXPOSE 8080

WORKDIR /app

# Commande pour lancer l'application
CMD ["conda", "run", "--no-capture-output", "-n", "dash_app", "python", "scripts/app.py"]
