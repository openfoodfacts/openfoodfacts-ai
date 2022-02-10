select
	la.id,
	la.index,
	la.bounding_box,
	la.annotation_value,
	la.annotation_value_tag,
	la.annotation_type,
	la.score,
	la.username,
	la.completed_at,
	la.taxonomy_value,
	ip.model_name,
	ip.model_version,
	image.barcode,
	image.source_image,
	image.server_domain
from
	logo_annotation la
join image_prediction ip on
	la.image_prediction_id = ip.id
join image on
    ip.image_id = image.id
where
	la.annotation_type is not null;