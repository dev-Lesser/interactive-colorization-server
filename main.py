from data import colorize_image as CI
import matplotlib.pyplot as plt
import numpy as np
from starlette.responses import StreamingResponse
import io, uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, File, UploadFile,Response, Form
from fastapi.responses import JSONResponse
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import cv2
from data import lab_gamut
from skimage import color
# Choose gpu to run the model on
import re
from colormath.color_objects import LabColor, sRGBColor 
from colormath.color_conversions import convert_color 

def hex_to_rgb(hx, hsl=False):
    if re.compile(r'#[a-fA-F0-9]{3}(?:[a-fA-F0-9]{3})?$').match(hx):
        div = 255.0 if hsl else 0
        if len(hx) <= 4:
            return tuple(int(hx[i]*2, 16) / div if div else
                            int(hx[i]*2, 16) for i in (1, 2, 3))
        return tuple(int(hx[i:i+2], 16) / div if div else
                        int(hx[i:i+2], 16) for i in (1, 3, 5))
    raise ValueError(f'"{hx}" is not a valid HEX code.')

gpu_id = -1

colorModel = CI.ColorizeImageTorch(Xd=256,maskcent=3)
colorModel.prep_net(path='./models/pytorch/pytorch.pth')

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Points(BaseModel):
    pointX: str
    pointY: str
    colors: str

def put_point(input_ab,mask,loc,p,val):
    # input_ab    2x256x256    current user ab input (will be updated)
    # mask        1x256x256    binary mask of current user input (will be updated)
    # loc         2 tuple      (h,w) of where to put the user input
    # p           scalar       half-patch size
    # val         2 tuple      (a,b) value of user input
    input_ab[:,loc[0]-p:loc[0]+p+1,loc[1]-p:loc[1]+p+1] = np.array(val)[:,np.newaxis,np.newaxis]
    mask[:,loc[0]-p:loc[0]+p+1,loc[1]-p:loc[1]+p+1] = 1
    return (input_ab,mask)

defaultFile = None
defaultImg = None
mask = np.zeros((1,256,256)) # giving no user points, so mask is all 0's
input_ab = np.zeros((2,256,256)) # ab values of user points, default to 0 for no input

@app.get("/api/v1/clear")
async def clear():

    global colorModel
    global defaultFile
    global defaultImg
    global mask
    global input_ab
    defaultFile = None
    defaultImg = None
    colorModel = CI.ColorizeImageTorch(Xd=256,maskcent=3)
    colorModel.prep_net(path='./models/pytorch/pytorch.pth')
    mask = np.zeros((1,256,256))
    input_ab = np.zeros((2,256,256))
    return JSONResponse(
            status_code=200,
            content='cleared'
        )

@app.post("/api/v1/colorize")
async def default_color_predict( file: bytes = File(...)):

    global defaultFile
    global defaultImg
    global colorModel
    global mask
    global input_ab
    defaultFile = file
    defaultImg = np.fromstring(file, dtype=np.uint8)
    defaultImg = cv2.imdecode(defaultImg, cv2.IMREAD_COLOR)

    colorModel.load_image(file) 

    img_out = colorModel.net_forward(input_ab,mask)

    img_out_fullres = colorModel.get_img_fullres()

    converted = img_out_fullres[...,::-1].copy()
    res, img_png = cv2.imencode(".png",converted)
    return StreamingResponse(io.BytesIO(img_png.tobytes()), media_type="image/png")

@app.post("/api/v1/colorize/point")
async def user_add_predict(
        pointsX: str = Form(...),pointsY: str= Form(...), colors: str= Form(...)
    ):
    global defaultFile
    global defaultImg
    global colorModel
    global mask
    global input_ab

    if (not defaultFile):
        return JSONResponse(
            status_code=404,
            content='error'
        )
    x_list      = [int(i) for i in pointsX.split(',')]
    y_list      = [int(i) for i in pointsY.split(',')]
    colors = colors.split(',')
    if (len(x_list) != len(y_list)) or(len(x_list)>3):
        return JSONResponse(
            status_code=404,
            content='error'
        )

    for x,y,_hex in list(zip(x_list,y_list,colors)):

        rgb = sRGBColor(hex_to_rgb(_hex)[0],hex_to_rgb(_hex)[1],hex_to_rgb(_hex)[2],  is_upscaled=True)
        lab =  convert_color(rgb, LabColor).get_value_tuple()
        _color = lab[1:]
        (input_ab,mask) = put_point(input_ab,mask,[x,y],3,_color)
        img_out = colorModel.net_forward(input_ab,mask) # run model, returns 256x256 image


    mask_fullres = colorModel.get_img_mask_fullres() # get input mask in full res
    img_in_fullres = colorModel.get_input_img_fullres() # get input image in full res
    img_out_fullres = colorModel.get_img_fullres() # get image at full resolution



    img_out_fullres = colorModel.get_img_fullres() # get image at full resolution
    converted = img_out_fullres[...,::-1].copy()

    res, img_png = cv2.imencode(".png", converted)
    return StreamingResponse(io.BytesIO(img_png.tobytes()), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
