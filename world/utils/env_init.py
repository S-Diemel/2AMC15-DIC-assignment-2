def create_delivery_zones(racks, width, height, margin):
    """
    Creates rectangular delivery zones around a list of storage racks, but make sure that the rectangular delivery areas 
    do not extend outside the environment boundaries.
    """
    delivery_zones = []
    
    # Helper function to process each potential zone
    def add_clipped_zone(xmin, ymin, xmax, ymax):
        """The created delivery zone rectangles are clipped such that do not extend outside the environment boundaries."""
        # Ensure delivery rectangle coordinates do not violate the warehouse dimensions (0, 0, width, height)
        clipped_xmin = max(0, xmin)
        clipped_ymin = max(0, ymin)
        clipped_xmax = min(width, xmax)
        clipped_ymax = min(height, ymax)

        # Only add the zone if it has a valid, positive area after clipping. This prevents zero-width/height delivery rectangles.
        if clipped_xmax > clipped_xmin and clipped_ymax > clipped_ymin:
            delivery_zones.append((clipped_xmin, clipped_ymin, clipped_xmax, clipped_ymax))

    # Iterate over the storage racks to create its surrounding zones
    for (xmin, ymin, xmax, ymax) in racks:
        # Calculate rectangle coordinates for all four rectangles along the storage racks
        
        add_clipped_zone(xmin - margin, ymax, xmax + margin, ymax + margin)  # Above
        add_clipped_zone(xmin - margin, ymin - margin, xmax + margin, ymin)  # Below
        add_clipped_zone(xmin - margin, ymin, xmin, ymax)  # Left
        add_clipped_zone(xmax, ymin, xmax + margin, ymax)  # Right
        
    return delivery_zones