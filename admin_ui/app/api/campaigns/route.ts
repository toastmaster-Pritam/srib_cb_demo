import { ddbDocClient } from "@/lib/dynamodb";
import { ScanCommand } from "@aws-sdk/lib-dynamodb";
import { NextResponse } from "next/server";

export async function GET() {
    try {
        const data = await ddbDocClient.send(
            new ScanCommand({
                TableName: "Campaigns",
                Limit: 5
            })
        );

        return NextResponse.json({
            campaigns: data.Items || []
        })

    } catch (error) {
        console.error("Campaign fetching failed:", error);
        return NextResponse.json({
            error: "Failed to fetch campaigns"
        }, {
            status: 500
        })

    }
}