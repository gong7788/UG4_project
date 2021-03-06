(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b1)
		(clear b1)
		(on-table b2)
		(clear b2)
		(on-table b3)
		(clear b3)
		(on-table b4)
		(clear b4)
		(on-table b5)
		(clear b5)
		(on-table b6)
		(clear b6)
		(on-table b7)
		(clear b7)
		(on-table b8)
		(clear b8)
		(on-table b9)
		(clear b9)
		(in-tower t0)
		(clear t0)
		(fuchsia b0)
		(pink b0)
		(yellow b1)
		(darkblue b2)
		(blue b2)
		(lightgoldenrodyellow b3)
		(yellow b3)
		(yellow b4)
		(navy b5)
		(blue b5)
		(blue b6)
		(mediumblue b7)
		(blue b7)
		(yellow b8)
		(midnightblue b9)
		(blue b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (darkviolet ?x)) (exists (?y) (and (orange ?y) (on ?x ?y))))) (forall (?y) (or (not (orange ?y)) (exists (?x) (and (yellow ?x) (on ?x ?y))))) (forall (?y) (or (not (orange ?y)) (exists (?x) (and (blue ?x) (on ?x ?y)))))))
)