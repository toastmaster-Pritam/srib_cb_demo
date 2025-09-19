import { S3Client } from "@aws-sdk/client-s3";

export const s3 = new S3Client({
    region: "eu-west-1",
    endpoint: "http://localhost:4566",
    forcePathStyle: true,
    credentials: {
        accessKeyId: "test",
        secretAccessKey: "test"
    }
})
