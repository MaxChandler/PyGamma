import pygamma as pg


def fid(sys, b0=123.23):
    sys.OmegaAdjust(b0)
    H = pg.Hcs(sys) + pg.HJ(sys)
    D = pg.Fm(sys)
    ACQ = pg.acquire1D(pg.gen_op(D), H, 0.1)
    sigma = pg.sigma_eq(sys)
    sigma0 = pg.Ixpuls(sys, sigma, 90.0)
    mx = pg.TTable1D(ACQ.table(sigma0))
    return mx