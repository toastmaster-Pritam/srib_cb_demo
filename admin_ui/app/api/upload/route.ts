import { s3 } from "@/lib/s3";
import { v4 as uuidv4 } from "uuid"
import { NextRequest, NextResponse } from "next/server";
import { PutObjectCommand } from "@aws-sdk/client-s3";
import { ddbDocClient } from "@/lib/dynamodb";
import { PutCommand } from "@aws-sdk/lib-dynamodb";
import { formatISO } from "date-fns";


export async function POST(req: NextRequest) {
    const formData = await req.formData();
    const file = formData.get("file") as File;

    if (!file) {
        return NextResponse.json({
            error: "No file uploaded!",
        },
            { status: 400 })
    }

    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    const bucketName = "local-campaigns"
    const key = `campaigns/${uuidv4()}_${Date.now()}-${file.name}`;


    try {
        await s3.send(new PutObjectCommand({
            Bucket: bucketName,
            Key: key,
            Body: buffer
        })
        );

        await ddbDocClient.send(new PutCommand({
            TableName: "Campaigns",
            Item: {
                campaignId: uuidv4(),
                name: file.name,
                uploadDate: formatISO(new Date()),
                key,
            }
        }))

        return NextResponse.json({
            success: true
        })


    }
    catch (err) {
        console.error("Upload error:", err);
        return NextResponse.json({
            error: "Upload failed!"
        }, { status: 500 })

    }


}