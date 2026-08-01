ALTER TABLE user_attributes
    ALTER COLUMN attribute_text TYPE text[] USING ARRAY[attribute_text];

ALTER TABLE user_attributes
    RENAME COLUMN attribute_text TO value;

ALTER TABLE user_attributes
    ALTER COLUMN attribute_type SET NOT NULL;
