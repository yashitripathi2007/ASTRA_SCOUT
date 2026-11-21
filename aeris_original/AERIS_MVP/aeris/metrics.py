def summarise(rover_route, aeris_route, rover_energy, aeris_energy):
    return {
        "rover_only_steps": len(rover_route) if rover_route else 0,
        "aeris_steps": len(aeris_route) if aeris_route else 0,
        "rover_only_energy": rover_energy,
        "aeris_total_energy": aeris_energy,
    }
