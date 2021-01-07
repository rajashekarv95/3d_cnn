def get_delta(bb1, bb2):
    dx = bb2[0] - bb1[0]
    dy = bb2[2] - bb1[2]
    
    dlx = (bb2[1] - bb2[0]) - (bb1[1] - bb1[0])
    dly = (bb2[3] - bb2[2]) - (bb1[3] - bb1[2])
    
    ret = [dx, dy, dlx, dly]
    return ret