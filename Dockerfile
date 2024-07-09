# Basic conda image
FROM continuumio/miniconda3

# New conda env
RUN conda create -n dash_app python=3.9 -y
SHELL ["conda", "run", "-n", "dash_app", "/bin/bash", "-c"]

# Copier le fichier des dépendances (environment.yml)
COPY environment.yml /tmp/environment.yml

# Installer les dépendances
RUN conda env update -n dash_app -f /tmp/environment.yml

# Activer l'environnement conda
RUN echo "conda activate dash_app" >> ~/.bashrc
SHELL ["conda", "run", "-n", "dash_app", "/bin/bash", "-c"]

# Copier le code de l'application
COPY . /app
WORKDIR /app

# Exposer le port sur lequel Dash va tourner
EXPOSE 8050

# Commande pour lancer l'application
CMD ["conda", "run", "--no-capture-output", "-n", "dash_app", "python", "app.py"]