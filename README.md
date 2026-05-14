# Docker Setup Guide

This guide explains how to activate and use Docker for the darko-peopleflow project.

## Prerequisites

Ensure you have Docker installed on your system.

## Activating Docker

Follow these steps to set up and enter the Docker environment:

### Step 1: Build the Docker Image
```bash
./docbuild.sh
```
This script builds the Docker image with all necessary dependencies.

### Step 2: Start the Docker Container
```bash
./docrun.sh
```
This script starts the Docker container.

### Step 3: Enter the Docker Shell
```bash
./docshell.sh
```
This script opens an interactive shell inside the Docker container.

### Step 4: Build and Source the ROS Workspace
```bash
catkin build
exit 
./docshell.sh
```
- `catkin build`: Builds the ROS workspace packages
- `exit`: Exits the current shell session
- `./docshell.sh`: Re-enters the Docker shell to source the updated ROS workspace

### Step 5: Stop the Docker Container
When you're done working:
```bash
exit
./docstop.sh
```
- `exit`: Exits the Docker shell
- `./docstop.sh`: Stops the Docker container

## Working Inside the Docker

Once you're inside the Docker shell (via `./docshell.sh`), you can use the following commands to manage the simulation:

### Start the Simulation
```bash
tstart
```

### Stop the Simulation
```bash
tstop
```

## Quick Reference

| Command | Purpose |
|---------|---------|
| `./docbuild.sh` | Build Docker image |
| `./docrun.sh` | Start Docker container |
| `./docshell.sh` | Enter Docker shell |
| `catkin build` | Build ROS workspace (inside Docker)|
| `exit` | Exit Docker (inside Docker)|
| `./docstop.sh` | Stop container |
| `tstart` | Start simulation (inside Docker) |
| `tstop` | Stop simulation (inside Docker) |

## Workflow Example

```bash
# Initial setup
./docbuild.sh (your pc)
./docrun.sh (your pc)
./docshell.sh (your pc)

# Inside Docker: build and exit
catkin build (docker)
exit (docker)
./docshell.sh (your pc)

# Inside Docker: run simulation
tstart
# ... do your work ...
tstop

# Exit and cleanup
exit (docker)
./docstop.sh (your pc)
```
