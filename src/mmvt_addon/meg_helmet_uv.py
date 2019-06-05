for v in bm.verts:
    uvs = uvs_from_vert(uv_lay, v)
    for uv in uvs:
        x0,y0 = uv
        f = symGauss2D(x0,y0,sigma)
        for i, x in enumerate(np.linspace(0,1,size)):
            for j, y in enumerate(np.linspace(0,1,size)):
                im[i,j] += f(x,y)