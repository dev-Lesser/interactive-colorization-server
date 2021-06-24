from data import colorize_image as CI
import numpy as np
from starlette.responses import StreamingResponse
import io, uvicorn
from fastapi import FastAPI,  File, Form
from fastapi.responses import JSONResponse
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
# Choose gpu to run the model on
import base64
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

mask = np.zeros((1,256,256)) 
input_ab = np.zeros((2,256,256)) 
colorModelDict = {}
@app.get("/api/v1/clear")
async def clear(timestamp: int):

    global colorModelDict
    global mask
    global input_ab
    if timestamp not in colorModelDict.keys():
        return JSONResponse(
            status_code=404,
            content='error'
        )
    del colorModelDict[str(timestamp)]

    mask = np.zeros((1,256,256))
    input_ab = np.zeros((2,256,256))
    return JSONResponse(
            status_code=200,
            content='cleared'
        )

@app.post("/api/v1/colorize")
async def default_color_predict(timestamp: int,file: bytes = File(...)):
    global colorModelDict
    global mask
    global input_ab
    colorModel = CI.ColorizeImageTorch(Xd=256,maskcent=3)
    colorModel.prep_net(path='./models/pytorch/pytorch.pth')

    mask = np.zeros((1,256,256)) 
    input_ab = np.zeros((2,256,256)) 

    colorModelDict[str(timestamp)] = colorModel
    colorModelDict[str(timestamp)].load_image(file)

    _ = colorModelDict[str(timestamp)].net_forward(input_ab,mask)

    img_out_fullres = colorModelDict[str(timestamp)].get_img_fullres()

    converted = img_out_fullres[...,::-1].copy()
    _, img_png = cv2.imencode(".png",converted)

    return base64.b64encode(img_png)
            

@app.post("/api/v1/colorize/point")
async def user_add_predict(timestamp: int,
        pointsX: str = Form(...),pointsY: str= Form(...), colors: str= Form(...)
    ):
    global colorModelDict
    global mask
    global input_ab
    print(colorModelDict)
    if str(timestamp) not in colorModelDict.keys():
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

        rgb = sRGBColor(hex_to_rgb(_hex)[0],hex_to_rgb(_hex)[1],hex_to_rgb(_hex)[2],  is_upscaled=True) # hex -> rgb
        lab =  convert_color(rgb, LabColor).get_value_tuple() # rgb -> lab
        _color = lab[1:] # lab -> ab
        (input_ab,mask) = put_point(input_ab,mask,[x,y],3,_color)
        img_out = colorModelDict[str(timestamp)].net_forward(input_ab,mask) # run model, returns 256x256 image


    _ = colorModelDict[str(timestamp)].get_img_mask_fullres() # get input mask in full res
    _ = colorModelDict[str(timestamp)].get_input_img_fullres() # get input image in full res
    img_out_fullres = colorModelDict[str(timestamp)].get_img_fullres() # get image at full resolution



    img_out_fullres = colorModelDict[str(timestamp)].get_img_fullres() # get image at full resolution
    converted = img_out_fullres[...,::-1].copy()

    _, img_png = cv2.imencode(".png", converted)

    # print(type(img_png))
    
    # result_img_string = img_png.tobytes()
    # print(base64.b64encode(img_png))
    return base64.b64encode(img_png)
    # return StreamingResponse(io.BytesIO(img_png.tobytes()), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
