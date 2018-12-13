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
		(lightgoldenrodyellow b0)
		(yellow b0)
		(lightyellow b1)
		(yellow b1)
		(bisque b2)
		(orange b2)
		(forestgreen b3)
		(green b3)
		(seagreen b4)
		(green b4)
		(darkred b5)
		(red b5)
		(cornflowerblue b6)
		(blue b6)
		(bisque b7)
		(orange b7)
		(darkorchid b8)
		(purple b8)
		(indigo b9)
		(purple b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?x) (or (not (pink ?x)) (exists (?y) (and (red ?y) (on ?x ?y))))) (forall (?y) (or (not (red ?y)) (exists (?x) (and (purple ?x) (on ?x ?y)))))))
)